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
import com.integrallis.vectors.studio.core.search.DocumentPageView;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.HashMap;
import java.util.Map;

/** Per-collection overview + paginated preview fragment. */
public final class CollectionRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;

  public CollectionRoutes(StudioSession session, ViewRenderer renderer) {
    this.session = session;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/collections/{name}", this::overview)
        .get("/collections/{name}/preview", this::preview);
  }

  private void overview(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    try {
      var summary = session.backend().describe(name);
      Map<String, Object> ctx = new HashMap<>();
      ctx.put("summary", summary);
      renderer.render(res, "collection.jte", ctx);
    } catch (IllegalArgumentException e) {
      res.status(Status.NOT_FOUND_404).send("collection not found: " + name);
    }
  }

  private void preview(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    int offset = req.query().first("offset").asOptional().map(Integer::parseInt).orElse(0);
    int limit = req.query().first("limit").asOptional().map(Integer::parseInt).orElse(25);
    if (limit < 1) limit = 1;
    if (limit > 200) limit = 200;
    if (offset < 0) offset = 0;
    DocumentPageView page = session.backend().documentPage(name, offset, limit);
    Map<String, Object> ctx =
        Map.of(
            "name", name,
            "page", page.items(),
            "offset", offset,
            "limit", limit,
            "total", page.total());
    renderer.renderFragment(res, Status.OK_200, "partials/documentList.jte", ctx);
  }
}
