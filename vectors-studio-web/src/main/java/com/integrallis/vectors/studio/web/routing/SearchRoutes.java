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

import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.List;
import java.util.Map;

/**
 * Search HTMX route — POST returns a hits-list fragment.
 *
 * <p>Turning a text query into a vector requires an embedding model, which the embedded Studio
 * backend does not have. Rather than fabricate a meaningless query vector (which would return
 * results unrelated to the query text), this returns a clear notice. Vector-space exploration in
 * the Studio is driven by clicking an existing point (search-by-id), which uses that point's real
 * vector.
 */
public final class SearchRoutes implements HttpService {

  private static final String NO_EMBEDDER_NOTE =
      "Text search requires an embedding model, which this Studio backend does not have. "
          + "Click a point to find its nearest neighbours instead.";

  private final ViewRenderer renderer;

  public SearchRoutes(ViewRenderer renderer) {
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.post("/collections/{name}/search", this::execute);
  }

  private void execute(ServerRequest req, ServerResponse res) {
    renderer.renderFragment(
        res,
        Status.OK_200,
        "partials/hitsList.jte",
        Map.of("hits", List.<SearchHit>of(), "note", NO_EMBEDDER_NOTE));
  }
}
