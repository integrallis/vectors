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

  private void upsert(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
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
    List<Document> docs = new ArrayList<>(body.documents().size());
    try {
      for (DocumentDto dto : body.documents()) {
        docs.add(dto.toDocument());
      }
    } catch (IllegalArgumentException e) {
      RouteSupport.sendProblem(
          res, Status.BAD_REQUEST_400, "invalid document", e.getMessage(), req);
      return;
    }
    VectorCollection c = col.get();
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
    }
    registry.bumpEpoch(name);
    RouteSupport.sendJson(res, Status.OK_200, new UpsertDocumentsResponse(docs.size(), c.size()));
  }

  private void delete(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
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
