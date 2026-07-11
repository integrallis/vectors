/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.studio.web.routing;

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.CollectionSummary;
import com.integrallis.vectors.studio.web.dataset.DatasetCatalog;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.List;
import java.util.Map;

/** Collections-list landing page plus collection-level destructive actions. */
public final class HomeRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;
  private final DatasetCatalog datasetCatalog;

  public HomeRoutes(StudioSession session, ViewRenderer renderer, DatasetCatalog datasetCatalog) {
    this.session = session;
    this.renderer = renderer;
    this.datasetCatalog = datasetCatalog;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/", this::redirectToCollections)
        .get("/collections", this::collections)
        .delete("/collections/{name}", this::deleteCollection);
  }

  private void redirectToCollections(ServerRequest req, ServerResponse res) {
    res.status(Status.MOVED_PERMANENTLY_301).header("Location", "/collections").send();
  }

  private void collections(ServerRequest req, ServerResponse res) {
    List<CollectionSummary> all = session.backend().listCollections();
    renderer.render(
        res, "collections.jte", Map.of("collections", all, "sampleIds", datasetCatalog.ids()));
  }

  private void deleteCollection(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    try {
      session.backend().deleteCollection(name);
    } catch (IllegalArgumentException e) {
      res.status(Status.NOT_FOUND_404).send("collection not found: " + name);
      return;
    } catch (RuntimeException e) {
      res.status(Status.INTERNAL_SERVER_ERROR_500).send(e.getMessage());
      return;
    }
    // HTMX swap-outerHTML on the row: empty body removes the <tr>.
    res.status(Status.OK_200).header("Content-Type", "text/html; charset=utf-8").send("");
  }
}
