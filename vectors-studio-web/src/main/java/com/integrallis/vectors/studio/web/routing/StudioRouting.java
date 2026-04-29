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
import com.integrallis.vectors.studio.web.view.JteEngineFactory;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import gg.jte.TemplateEngine;
import io.helidon.webserver.http.HttpRouting;
import io.helidon.webserver.staticcontent.StaticContentService;

/** Wires every Studio HTTP route, JTE renderer, and static-asset handler. */
public final class StudioRouting {

  private final StudioSession session;
  private final ProjectionJobManager jobs;
  private final ViewRenderer renderer;

  public StudioRouting(StudioSession session, ProjectionJobManager jobs) {
    this.session = session;
    this.jobs = jobs;
    TemplateEngine engine = JteEngineFactory.create();
    this.renderer = new ViewRenderer(engine);
  }

  /** Applies all Studio routes to {@code rules}. */
  @SuppressWarnings("removal")
  public void apply(HttpRouting.Builder rules) {
    rules
        .register("/static", StaticContentService.create("/static"))
        .register(new HomeRoutes(session, renderer))
        .register(new CollectionRoutes(session, renderer))
        .register(new SearchRoutes(session, renderer))
        .register(new DocumentRoutes(session, renderer))
        .register(new ProjectorRoutes(session, jobs, renderer))
        .register(new RecommenderRoutes(session, renderer))
        .register(new ApiRoutes(session, jobs))
        .register(new HealthRoutes());
  }
}
