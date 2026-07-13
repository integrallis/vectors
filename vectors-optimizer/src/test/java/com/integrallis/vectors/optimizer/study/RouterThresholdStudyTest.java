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
package com.integrallis.vectors.optimizer.study;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.sampler.RandomSampler;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.router.EmbeddingFunction;
import com.integrallis.vectors.router.Route;
import com.integrallis.vectors.router.SemanticRouter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RouterThresholdStudyTest {

  // Three semantically separated 3-D clusters: sports along x, food along y, weather along z.
  private static final Map<String, float[]> EMB = new HashMap<>();

  static {
    String[] sports = {"football", "basketball", "soccer", "tennis", "hockey"};
    String[] food = {"pasta", "sushi", "bread", "salad", "pizza"};
    String[] weather = {"rain", "snow", "wind", "sun", "cloud"};
    for (String t : sports) EMB.put(t, new float[] {1.0f, 0.05f, 0.0f});
    for (String t : food) EMB.put(t, new float[] {0.05f, 1.0f, 0.0f});
    for (String t : weather) EMB.put(t, new float[] {0.0f, 0.05f, 1.0f});
    EMB.put("noise-1", new float[] {0.4f, 0.4f, 0.4f}); // off all clusters
    EMB.put("noise-2", new float[] {0.5f, 0.5f, 0.5f});
    EMB.put("noise-3", new float[] {0.3f, 0.3f, 0.3f});
  }

  private static final EmbeddingFunction EMBEDDER = text -> EMB.get(text);

  private static List<LabeledQuery> labeledProbes() {
    List<LabeledQuery> out = new ArrayList<>();
    for (String t :
        List.of(
            "football",
            "basketball",
            "soccer",
            "tennis",
            "hockey",
            "football",
            "basketball",
            "soccer",
            "tennis",
            "hockey")) {
      out.add(LabeledQuery.routerProbe(t, "sports"));
    }
    for (String t :
        List.of(
            "pasta", "sushi", "bread", "salad", "pizza", "pasta", "sushi", "bread", "salad",
            "pizza")) {
      out.add(LabeledQuery.routerProbe(t, "food"));
    }
    for (String t :
        List.of("rain", "snow", "wind", "sun", "cloud", "rain", "snow", "wind", "sun", "cloud")) {
      out.add(LabeledQuery.routerProbe(t, "weather"));
    }
    // 30 expected-route + 0 expected-miss is a deliberate single-class-only setup; total=30.
    return out;
  }

  private static RouterThresholdStudy.RouterFactory factoryWithRoutes() {
    Route sports =
        Route.builder()
            .name("sports")
            .references(List.of("football", "basketball"))
            .distanceThreshold(0.5)
            .build();
    Route food =
        Route.builder()
            .name("food")
            .references(List.of("pasta", "sushi"))
            .distanceThreshold(0.5)
            .build();
    Route weather =
        Route.builder()
            .name("weather")
            .references(List.of("rain", "snow"))
            .distanceThreshold(0.5)
            .build();
    return thresholds ->
        new SemanticRouter(
            EMBEDDER,
            List.of(
                rebuild(sports, thresholds),
                rebuild(food, thresholds),
                rebuild(weather, thresholds)));
  }

  private static Route rebuild(Route base, Map<String, Double> thresholds) {
    Double t = thresholds.get(base.getName());
    return Route.builder()
        .name(base.getName())
        .references(base.getReferences())
        .distanceThreshold(t != null ? t : base.getDistanceThreshold())
        .build();
  }

  @Test
  void threeRouteRouterReachesF1AboveThreshold() {
    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.DoubleRange("threshold_sports", 0.05, 0.95, false),
                new ParamSpec.DoubleRange("threshold_food", 0.05, 0.95, false),
                new ParamSpec.DoubleRange("threshold_weather", 0.05, 0.95, false)));

    RouterThresholdStudy study = new RouterThresholdStudy(factoryWithRoutes(), labeledProbes());
    RandomSampler sampler = new RandomSampler(space, 17L);

    double bestAccuracy = 0.0;
    for (int i = 0; i < 20; i++) {
      Trial trial = sampler.next(List.of());
      TrialResult tr = study.runOne(trial);
      if (tr.objectiveScore() > bestAccuracy) bestAccuracy = tr.objectiveScore();
    }
    assertThat(bestAccuracy).isGreaterThanOrEqualTo(0.8);
  }

  @Test
  void objectiveScoreHonorsWeights_notHardcodedAccuracy() {
    // Regression (audit optimizer #20): the study hardcoded objectiveScore=accuracy and never
    // called Objective.score(...), so configured latency/cost/quality weights were silent no-ops.
    RouterThresholdStudy.RouterFactory factory = factoryWithRoutes();
    List<LabeledQuery> probes = labeledProbes();
    Map<String, Object> params = new HashMap<>();
    params.put("threshold_sports", 0.5);
    params.put("threshold_food", 0.5);
    params.put("threshold_weather", 0.5);
    Trial trial = new Trial("t-0", params);

    // Default (recall-only) weights preserve the historical behaviour: objective == accuracy.
    TrialResult base = new RouterThresholdStudy(factory, probes).runOne(trial);
    assertThat(base.recallAtK()).as("fixture must yield non-trivial accuracy").isGreaterThan(0.0);
    assertThat(base.objectiveScore()).isEqualTo(base.recallAtK());

    // Doubling the recall weight must double the composite — a deterministic proof the weights are
    // actually applied. A hardcoded objectiveScore=accuracy would ignore this.
    ObjectiveWeights doubled = ObjectiveWeights.builder().recallWeight(2.0).build();
    TrialResult w = new RouterThresholdStudy(factory, probes, doubled).runOne(trial);
    assertThat(w.recallAtK()).isEqualTo(base.recallAtK());
    assertThat(w.objectiveScore()).isEqualTo(2.0 * base.recallAtK());

    // A latency weight (a MIN cost axis) can only reduce, never increase, the composite.
    ObjectiveWeights withLatency =
        ObjectiveWeights.builder()
            .recallWeight(1.0)
            .latencyP95Weight(0.5)
            .latencyP95ReferenceUs(1.0)
            .build();
    TrialResult wl = new RouterThresholdStudy(factory, probes, withLatency).runOne(trial);
    assertThat(wl.objectiveScore()).isLessThanOrEqualTo(base.objectiveScore());
  }
}
