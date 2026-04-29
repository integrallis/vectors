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
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.List;
import java.util.Map;

/** Home/collections-list landing page. */
public final class HomeRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;

  public HomeRoutes(StudioSession session, ViewRenderer renderer) {
    this.session = session;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/", this::home).get("/collections", this::collections);
  }

  private void home(ServerRequest req, ServerResponse res) {
    List<CollectionSummary> collections = session.backend().listCollections();
    renderer.render(res, "home.jte", Map.of("collections", collections));
  }

  private void collections(ServerRequest req, ServerResponse res) {
    List<CollectionSummary> all = session.backend().listCollections();
    renderer.render(res, "collections.jte", Map.of("collections", all));
  }
}
