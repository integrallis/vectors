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

import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.server.CollectionRegistry;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Objects;
import java.util.Optional;

/**
 * HTTP route for retrieving blobs (e.g. images) from the text index.
 *
 * <p>{@code GET /v1/collections/{name}/blobs/{docId}} returns the raw binary blob stored alongside
 * a document in the text index, or 404 if not found.
 */
public final class BlobRoutes implements HttpService {

  private final CollectionRegistry registry;

  public BlobRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/v1/collections/{name}/blobs/{docId}", this::getBlob);
  }

  private void getBlob(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    String docId = req.path().pathParameters().get("docId");
    if (!RouteSupport.validateName(name, req, res)) return;

    Optional<TextIndexSpi> textIndex = registry.getTextIndex(name);
    if (textIndex.isEmpty()) {
      RouteSupport.sendProblem(
          res, Status.NOT_FOUND_404, "no text index", "text index not available for " + name, req);
      return;
    }

    Optional<byte[]> blob = textIndex.get().getBlob(docId);
    if (blob.isEmpty()) {
      RouteSupport.sendProblem(
          res, Status.NOT_FOUND_404, "blob not found", "no blob for document " + docId, req);
      return;
    }

    res.status(Status.OK_200);
    res.header("Content-Type", "application/octet-stream");
    res.send(blob.get());
  }
}
