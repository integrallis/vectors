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
import com.integrallis.vectors.studio.core.search.DocumentView;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Map;

/** Per-document inspector page. */
public final class DocumentRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;

  public DocumentRoutes(StudioSession session, ViewRenderer renderer) {
    this.session = session;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/collections/{name}/documents/{id}", this::detail);
  }

  private void detail(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    String id = req.path().pathParameters().get("id");
    DocumentView doc = session.backend().getDocument(name, id);
    if (doc == null) {
      res.status(Status.NOT_FOUND_404).send("document not found: " + id);
      return;
    }
    renderer.render(res, "document.jte", Map.of("name", name, "doc", doc));
  }
}
