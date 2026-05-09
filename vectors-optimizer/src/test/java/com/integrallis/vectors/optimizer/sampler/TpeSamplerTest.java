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
package com.integrallis.vectors.optimizer.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.ScoredTrial;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TpeSamplerTest {

  // Continuous-only space large enough that random's blanket coverage is poor at modest budgets.
  private static final SearchSpace QUAD_SPACE =
      new SearchSpace(
          List.of(
              new ParamSpec.DoubleRange("x", -100.0, 100.0, false),
              new ParamSpec.DoubleRange("y", -100.0, 100.0, false)));

  // Higher is better; optimum at (3, -7) with score 0.
  private static double score(double x, double y) {
    double dx = x - 3.0;
    double dy = y + 7.0;
    return -(dx * dx + dy * dy);
  }

  private static double scoreOf(Trial t) {
    return score(t.getDouble("x"), t.getDouble("y"));
  }

  // A small integer grid used by the gamma-quantile split test.
  private static final SearchSpace GRID_SPACE =
      new SearchSpace(
          List.of(
              new ParamSpec.IntRange("x", -10, 10, false),
              new ParamSpec.Categorical("c", List.of("A", "B", "C"))));

  private static double gridScore(int x, String c) {
    return -((x - 3.0) * (x - 3.0) + (c.equals("B") ? 0.0 : 5.0));
  }

  @Test
  void delegatesToRandomBeforeStartup() {
    long seed = 99L;
    var hp = new TpeSampler.Hyperparameters(10, 24, 0.25, 1.0);
    TpeSampler tpe = new TpeSampler(GRID_SPACE, seed, hp);
    RandomSampler rnd = new RandomSampler(GRID_SPACE, seed);
    for (int i = 0; i < hp.nStartupTrials(); i++) {
      Trial a = tpe.next(List.of());
      Trial b = rnd.next(List.of());
      assertThat(a.params()).isEqualTo(b.params());
    }
  }

  @Test
  void outperformsRandomOnQuadratic() {
    int budget = 300;
    long[] seeds = {1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L};

    double tpeAvg = 0.0;
    double rndAvg = 0.0;
    for (long seed : seeds) {
      tpeAvg += runAndScore(new TpeSampler(QUAD_SPACE, seed), budget, true);
      rndAvg += runAndScore(new RandomSampler(QUAD_SPACE, seed), budget, false);
    }
    tpeAvg /= seeds.length;
    rndAvg /= seeds.length;

    // Averaged over a panel of seeds, TPE should beat random search on a 2D continuous landscape
    // and the gap should be material (>=5 score-units).
    assertThat(tpeAvg).isGreaterThan(rndAvg + 5.0);
  }

  private static double runAndScore(ParamSampler sampler, int budget, boolean feedHistory) {
    List<ScoredTrial> history = new ArrayList<>();
    double best = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < budget; i++) {
      Trial t = sampler.next(history);
      double s = scoreOf(t);
      best = Math.max(best, s);
      if (feedHistory) {
        history.add(new StubScoredTrial(t, s));
      }
    }
    return best;
  }

  @Test
  void splitsHistoryAtGammaQuantile() {
    // Construct a clearly bimodal history: 5 good (x near 3, c=B) and 15 bad (x=-10, c=A).
    long seed = 7L;
    var hp = new TpeSampler.Hyperparameters(0, 24, 0.25, 1.0);
    TpeSampler tpe = new TpeSampler(GRID_SPACE, seed, hp);

    List<ScoredTrial> history = new ArrayList<>();
    for (int i = 0; i < 5; i++) {
      Trial t = new Trial("g-" + i, java.util.Map.of("x", 3, "c", "B"));
      history.add(new StubScoredTrial(t, gridScore(3, "B")));
    }
    for (int i = 0; i < 15; i++) {
      Trial t = new Trial("b-" + i, java.util.Map.of("x", -10, "c", "A"));
      history.add(new StubScoredTrial(t, gridScore(-10, "A")));
    }

    int nearOptimum = 0;
    int batches = 30;
    for (int i = 0; i < batches; i++) {
      Trial t = tpe.next(history);
      if (Math.abs(t.getInt("x") - 3) <= 2 && t.getString("c").equals("B")) {
        nearOptimum++;
      }
    }
    // With clean bimodal evidence, most candidates should land near x=3, c=B.
    assertThat(nearOptimum).isGreaterThan(batches / 2);
  }

  @Test
  void handlesAllCategoricalSpace() {
    SearchSpace cat =
        new SearchSpace(
            List.of(
                new ParamSpec.Categorical("metric", List.of("COSINE", "DOT", "EUCLIDEAN")),
                new ParamSpec.Categorical("quantizer", List.of("NONE", "SQ8", "PQ"))));
    TpeSampler s = new TpeSampler(cat, 11L);
    List<ScoredTrial> history = new ArrayList<>();
    for (int i = 0; i < 30; i++) {
      Trial t = s.next(history);
      assertThat(t.getString("metric")).isIn("COSINE", "DOT", "EUCLIDEAN");
      assertThat(t.getString("quantizer")).isIn("NONE", "SQ8", "PQ");
      // Synthetic score: COSINE+SQ8 is best.
      double s2 = (t.getString("metric").equals("COSINE") ? 1.0 : 0.0)
          + (t.getString("quantizer").equals("SQ8") ? 1.0 : 0.0);
      history.add(new StubScoredTrial(t, s2));
    }
  }

  @Test
  void numericallyStableWithDuplicateObservations() {
    long seed = 3L;
    TpeSampler s = new TpeSampler(GRID_SPACE, seed);
    List<ScoredTrial> history = new ArrayList<>();
    // 20 identical observations
    for (int i = 0; i < 20; i++) {
      Trial t = new Trial("dup-" + i, java.util.Map.of("x", 0, "c", "A"));
      history.add(new StubScoredTrial(t, gridScore(0, "A")));
    }
    // Should not throw, NaN, or infinite-loop the sampler.
    for (int i = 0; i < 5; i++) {
      Trial t = s.next(history);
      assertThat(t.getInt("x")).isBetween(-10, 10);
      assertThat(t.getString("c")).isIn("A", "B", "C");
    }
  }
}
