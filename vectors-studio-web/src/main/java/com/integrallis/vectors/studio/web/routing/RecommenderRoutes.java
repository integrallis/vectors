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
import com.integrallis.vectors.studio.core.recommender.DatasetStatsCollector;
import com.integrallis.vectors.studio.core.recommender.ProjectionRecommendation;
import com.integrallis.vectors.studio.web.view.ViewRenderer;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.HashMap;
import java.util.Map;

/** "Recommend a projection" routes — heuristic always; LLM enrichment when configured. */
public final class RecommenderRoutes implements HttpService {

  private final StudioSession session;
  private final ViewRenderer renderer;

  public RecommenderRoutes(StudioSession session, ViewRenderer renderer) {
    this.session = session;
    this.renderer = renderer;
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/collections/{name}/recommend", this::recommend)
        .get("/collections/{name}/recommend/explain", this::explain);
  }

  private void recommend(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    var stats = DatasetStatsCollector.analyze(session.backend(), name, 1024);
    ProjectionRecommendation rec = session.heuristic().recommend(stats, 2);
    Map<String, Object> ctx = new HashMap<>();
    ctx.put("stats", stats);
    ctx.put("recommendation", rec);
    ctx.put("name", name);
    ctx.put("llmAvailable", session.llm().isPresent());
    renderer.render(res, "recommender.jte", ctx);
  }

  private void explain(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (session.llm().isEmpty()) {
      renderer.renderFragment(
          res,
          Status.OK_200,
          "partials/llmExplanation.jte",
          Map.of("explanation", "LLM enrichment not configured."));
      return;
    }
    var stats = DatasetStatsCollector.analyze(session.backend(), name, 1024);
    ProjectionRecommendation base = session.heuristic().recommend(stats, 2);
    ProjectionRecommendation enriched;
    try {
      enriched = session.llm().get().enrich(stats, base);
    } catch (RuntimeException e) {
      renderer.renderFragment(
          res,
          Status.OK_200,
          "partials/llmExplanation.jte",
          Map.of("explanation", "LLM call failed: " + e.getMessage()));
      return;
    }
    renderer.renderFragment(
        res,
        Status.OK_200,
        "partials/llmExplanation.jte",
        Map.of("explanation", enriched.llmExplanation()));
  }
}
