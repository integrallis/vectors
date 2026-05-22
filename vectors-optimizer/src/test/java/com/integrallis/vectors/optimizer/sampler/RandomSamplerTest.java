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
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RandomSamplerTest {

  @Test
  void respectsBounds() {
    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.IntRange("m", 8, 64, false),
                new ParamSpec.DoubleRange("alpha", 0.5, 1.5, false),
                new ParamSpec.Categorical("metric", List.of("COSINE", "DOT"))));
    RandomSampler s = new RandomSampler(space, 42L);
    for (int i = 0; i < 200; i++) {
      Trial t = s.next(List.of());
      int m = t.getInt("m");
      double a = t.getDouble("alpha");
      String metric = t.getString("metric");
      assertThat(m).isBetween(8, 64);
      assertThat(a).isBetween(0.5, 1.5);
      assertThat(metric).isIn("COSINE", "DOT");
    }
  }

  @Test
  void logScaleRangesProduceLogUniform() {
    SearchSpace space =
        new SearchSpace(List.of(new ParamSpec.DoubleRange("ef", 1.0, 1000.0, true)));
    RandomSampler s = new RandomSampler(space, 7L);
    int below10 = 0;
    int total = 1000;
    for (int i = 0; i < total; i++) {
      double v = s.next(List.of()).getDouble("ef");
      if (v < 10.0) below10++;
    }
    // Log-uniform over [1, 1000]: P(v < 10) = log(10)/log(1000) = 1/3.
    assertThat((double) below10 / total)
        .isCloseTo(1.0 / 3.0, org.assertj.core.data.Offset.offset(0.05));
  }

  @Test
  void seedYieldsDeterministicSequence() {
    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.IntRange("m", 8, 64, false),
                new ParamSpec.Categorical("metric", List.of("COSINE", "DOT"))));
    RandomSampler a = new RandomSampler(space, 99L);
    RandomSampler b = new RandomSampler(space, 99L);
    for (int i = 0; i < 20; i++) {
      assertThat(a.next(List.of()).params()).isEqualTo(b.next(List.of()).params());
    }
  }

  @Test
  void convergesOnSyntheticQuadratic() {
    // Large-budget random search on a tiny landscape converges to the optimum.
    SearchSpace space = new SearchSpace(List.of(new ParamSpec.IntRange("x", -10, 10, false)));
    RandomSampler s = new RandomSampler(space, 1L);
    int bestX = Integer.MAX_VALUE;
    for (int i = 0; i < 200; i++) {
      int x = s.next(List.of()).getInt("x");
      if (Math.abs(x - 3) < Math.abs(bestX - 3)) bestX = x;
    }
    assertThat(bestX).isEqualTo(3);
  }
}
