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
import com.integrallis.vectors.studio.web.projection.ProjectionJobManager;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Map;

/** Renders the 3D projector page; the JS island talks back via {@link ApiRoutes}. */
public final class ProjectorRoutes implements HttpService {

  private final StudioSession session;
  private final ProjectionJobManager jobs;
  private final ViewRenderer renderer;

  public ProjectorRoutes(StudioSession session, ProjectionJobManager jobs, ViewRenderer renderer) {
    this.session = session;
    this.jobs = jobs;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/collections/{name}/projector", this::page);
  }

  private void page(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    var summary = session.backend().describe(name);
    renderer.render(res, "projector.jte", Map.of("summary", summary));
  }
}
