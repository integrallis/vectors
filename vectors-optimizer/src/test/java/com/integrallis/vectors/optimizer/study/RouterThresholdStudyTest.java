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
}
