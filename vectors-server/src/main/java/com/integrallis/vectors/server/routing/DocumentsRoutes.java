/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.dto.DocumentDto;
import com.integrallis.vectors.server.dto.UpsertDocumentsRequest;
import com.integrallis.vectors.server.dto.UpsertDocumentsResponse;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HTTP routes for document ingestion and deletion on a collection.
 *
 * <p>{@code POST /v1/collections/{name}/documents} upserts every document in the batch under the
 * collection's per-name write lock and commits a single new generation. {@code DELETE
 * /v1/collections/{name}/documents/{id}} stages a tombstone and commits.
 */
public final class DocumentsRoutes implements HttpService {

  private static final Logger LOG = LoggerFactory.getLogger(DocumentsRoutes.class);

  private final CollectionRegistry registry;

  /**
   * @param registry shared registry of open collections
   */
  public DocumentsRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .post("/v1/collections/{name}/documents", this::upsert)
        .delete("/v1/collections/{name}/documents/{id}", this::delete)
        .post("/v1/collections/{name}/commit", this::commit);
  }

  /** Maximum documents per upsert batch. Prevents unbounded memory allocation from user input. */
  private static final int MAX_BATCH_SIZE = 10_000;

  private void upsert(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    UpsertDocumentsRequest body;
    try {
      body =
          RouteSupport.MAPPER.readValue(req.content().inputStream(), UpsertDocumentsRequest.class);
    } catch (Exception e) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "malformed request body", e.getMessage(), req);
      return;
    }
    if (body.documents().isEmpty()) {
      RouteSupport.sendJson(res, Status.OK_200, new UpsertDocumentsResponse(0, col.get().size()));
      return;
    }
    if (body.documents().size() > MAX_BATCH_SIZE) {
      RouteSupport.sendProblem(
          res,
          Status.BAD_REQUEST_400,
          "batch too large",
          "max " + MAX_BATCH_SIZE + " documents per request, got " + body.documents().size(),
          req);
      return;
    }
    VectorCollection c = col.get();
    int expectedDim = c.config().dimension();
    List<Document> docs = new ArrayList<>(body.documents().size());
    try {
      for (DocumentDto dto : body.documents()) {
        Document d = dto.toDocument();
        if (d.vector().length != expectedDim) {
          throw new IllegalArgumentException(
              "vector dimension mismatch for id='"
                  + d.id()
                  + "': expected "
                  + expectedDim
                  + ", got "
                  + d.vector().length);
        }
        docs.add(d);
      }
    } catch (IllegalArgumentException e) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "invalid document", e.getMessage(), req);
      return;
    }
    try {
      for (Document d : docs) {
        c.upsert(d);
      }
      c.commit();
    } catch (IllegalArgumentException e) {
      LOG.debug("rejected upsert batch for '{}': {}", name, e.getMessage());
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "invalid document batch", e.getMessage(), req);
      return;
    } catch (RuntimeException e) {
      LOG.error("commit failed for '{}': {}", name, e.getMessage(), e);
      RouteSupport.sendProblem(
          res,
          Status.INTERNAL_SERVER_ERROR_500,
          "commit failed",
          "upsert may not have been persisted",
          req);
      return;
    }
    registry.bumpEpoch(name);
    RouteSupport.sendJson(res, Status.OK_200, new UpsertDocumentsResponse(docs.size(), c.size()));
  }

  private void delete(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    String id = req.path().pathParameters().get("id");
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    VectorCollection c = col.get();
    if (!c.contains(id)) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "document not found", id, req);
      return;
    }
    boolean deleted = c.delete(id);
    if (deleted) {
      c.commit();
      registry.bumpEpoch(name);
    }
    res.status(Status.NO_CONTENT_204).send();
  }

  private void commit(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    col.get().commit();
    registry.bumpEpoch(name);
    res.status(Status.NO_CONTENT_204).send();
  }
}
