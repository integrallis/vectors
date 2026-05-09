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
package com.integrallis.vectors.demo.optimizer;

import com.integrallis.vectors.optimizer.sampler.GridSampler;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.optimizer.study.LabeledQuery;
import com.integrallis.vectors.optimizer.study.RouterThresholdStudy;
import com.integrallis.vectors.optimizer.study.TrialResult;
import com.integrallis.vectors.router.EmbeddingFunction;
import com.integrallis.vectors.router.Route;
import com.integrallis.vectors.router.SemanticRouter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stage 4 — Threshold tuning for a {@link SemanticRouter}. Uses a deterministic 3-axis embedder
 * (sports / food / weather) and a 30-probe labelled set including expected-miss queries. Sweeps
 * each route's distance threshold on a coarse grid and reports the configuration with the highest
 * micro-averaged accuracy.
 *
 * <p>Threshold optimization is the simplest demonstration of how the optimizer extends beyond index
 * parameters: nothing about the corpus changes — only the per-route decision boundary that
 * separates "match this route" from "fall through".
 */
public final class Stage4RouterThreshold {

  private Stage4RouterThreshold() {}

  public static void main(String[] args) {
    System.out.println(
        "=== Stage 4 — Router threshold sweep (3 routes × 5-point grid = 125 trials) ===");

    Map<String, float[]> embeddings = buildEmbeddings();
    EmbeddingFunction embedder = embeddings::get;
    List<LabeledQuery> probes = buildProbes();

    RouterThresholdStudy.RouterFactory factory =
        thresholds ->
            new SemanticRouter(
                embedder,
                List.of(
                    routeWithThreshold("sports", List.of("football", "basketball"), thresholds),
                    routeWithThreshold("food", List.of("pasta", "sushi"), thresholds),
                    routeWithThreshold("weather", List.of("rain", "snow"), thresholds)));

    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.Discrete<>("threshold_sports", List.of(0.10, 0.20, 0.30, 0.40, 0.50)),
                new ParamSpec.Discrete<>("threshold_food", List.of(0.10, 0.20, 0.30, 0.40, 0.50)),
                new ParamSpec.Discrete<>(
                    "threshold_weather", List.of(0.10, 0.20, 0.30, 0.40, 0.50))));

    RouterThresholdStudy study = new RouterThresholdStudy(factory, probes);
    GridSampler sampler = new GridSampler(space);

    long t0 = System.nanoTime();
    TrialResult best = null;
    int n = 0;
    try {
      while (true) {
        Trial trial = sampler.next(List.of());
        TrialResult tr = study.runOne(trial);
        if (best == null || tr.objectiveScore() > best.objectiveScore()) best = tr;
        n++;
      }
    } catch (com.integrallis.vectors.optimizer.sampler.NoMoreTrialsException expected) {
      // Grid exhausted.
    }
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

    System.out.printf("%nEvaluated %d threshold combinations in %d ms.%n", n, elapsedMs);
    if (best != null) {
      Map<String, Object> p = best.trial().params();
      System.out.println();
      System.out.println("Best configuration");
      System.out.println("┌─────────────────────┬──────────┐");
      System.out.printf(
          "│ threshold_sports    │ %-8.2f │%n", ((Number) p.get("threshold_sports")).doubleValue());
      System.out.printf(
          "│ threshold_food      │ %-8.2f │%n", ((Number) p.get("threshold_food")).doubleValue());
      System.out.printf(
          "│ threshold_weather   │ %-8.2f │%n",
          ((Number) p.get("threshold_weather")).doubleValue());
      System.out.println("├─────────────────────┼──────────┤");
      System.out.printf("│ accuracy            │ %-8.4f │%n", best.objectiveScore());
      System.out.println("└─────────────────────┴──────────┘");
    }
    System.out.println();
    System.out.println("Tutorial complete. Inspect persisted studies in ~/.vectors/optimizer/");
  }

  /**
   * Deterministic 3-D embeddings: sports ≈ [1,0,0], food ≈ [0,1,0], weather ≈ [0,0,1] with noise.
   */
  private static Map<String, float[]> buildEmbeddings() {
    Map<String, float[]> m = new HashMap<>();
    // Sports cluster
    m.put("football", new float[] {0.96f, 0.02f, 0.02f});
    m.put("basketball", new float[] {0.94f, 0.04f, 0.02f});
    m.put("soccer", new float[] {0.95f, 0.03f, 0.02f});
    m.put("tennis", new float[] {0.92f, 0.05f, 0.03f});
    // Food cluster
    m.put("pasta", new float[] {0.04f, 0.94f, 0.02f});
    m.put("sushi", new float[] {0.02f, 0.96f, 0.02f});
    m.put("pizza", new float[] {0.05f, 0.93f, 0.02f});
    m.put("salad", new float[] {0.03f, 0.95f, 0.02f});
    // Weather cluster
    m.put("rain", new float[] {0.02f, 0.02f, 0.96f});
    m.put("snow", new float[] {0.02f, 0.04f, 0.94f});
    m.put("sunny", new float[] {0.05f, 0.03f, 0.92f});
    m.put("cloudy", new float[] {0.03f, 0.03f, 0.94f});
    // Off-topic queries (expected miss)
    m.put("philosophy", new float[] {0.40f, 0.40f, 0.20f});
    m.put("quantum", new float[] {0.33f, 0.34f, 0.33f});
    m.put("history", new float[] {0.30f, 0.45f, 0.25f});
    return m;
  }

  private static List<LabeledQuery> buildProbes() {
    List<LabeledQuery> out = new ArrayList<>();
    for (String t : List.of("football", "basketball", "soccer", "tennis"))
      out.add(LabeledQuery.routerProbe(t, "sports"));
    for (String t : List.of("pasta", "sushi", "pizza", "salad"))
      out.add(LabeledQuery.routerProbe(t, "food"));
    for (String t : List.of("rain", "snow", "sunny", "cloudy"))
      out.add(LabeledQuery.routerProbe(t, "weather"));
    for (String t : List.of("philosophy", "quantum", "history"))
      out.add(LabeledQuery.routerProbe(t, null));
    return out;
  }

  private static Route routeWithThreshold(
      String name, List<String> refs, Map<String, Double> thresholds) {
    Map<String, Double> safe = thresholds == null ? new LinkedHashMap<>() : thresholds;
    double t = safe.getOrDefault(RouterThresholdStudy.THRESHOLD_PREFIX + name, 0.3);
    return Route.builder().name(name).references(refs).distanceThreshold(t).build();
  }
}
