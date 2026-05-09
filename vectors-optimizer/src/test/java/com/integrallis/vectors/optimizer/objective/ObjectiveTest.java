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
package com.integrallis.vectors.optimizer.objective;

import static org.assertj.core.api.Assertions.assertThat;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ObjectiveTest {

  @Test
  void scoreSubtractsMinimizedAxes() {
    // recall=1.0 weighted 1.0 → +1.0
    // latency=1000us against ref=1000us, weight 0.5 → -0.5
    // Expected: 0.5
    ObjectiveWeights w =
        ObjectiveWeights.builder()
            .recallWeight(1.0)
            .latencyP95Weight(0.5)
            .latencyP95ReferenceUs(1_000.0)
            .build();
    double s = Objective.score(1.0, 0, 0, 0, 0, 1_000.0, 0L, 0L, w);
    assertThat(s).isCloseTo(0.5, Offset.offset(1e-9));
  }

  @Test
  void scoreIsZeroWhenAllWeightsZero() {
    ObjectiveWeights w =
        ObjectiveWeights.builder()
            .recallWeight(0)
            .ndcgWeight(0)
            .precisionWeight(0)
            .f1Weight(0)
            .mrrWeight(0)
            .build();
    double s = Objective.score(1.0, 1.0, 1.0, 1.0, 1.0, 5_000.0, 60_000L, 1_000_000L, w);
    assertThat(s).isZero();
  }

  @Test
  void scoreNormalizesLatencyToReference() {
    // recall=1.0 weight 1.0 → +1.0
    // latency=2x reference, clipped to 1.0, weight 1.0 → -1.0  → total 0.0
    ObjectiveWeights w =
        ObjectiveWeights.builder()
            .recallWeight(1.0)
            .latencyP95Weight(1.0)
            .latencyP95ReferenceUs(1_000.0)
            .build();
    double s = Objective.score(1.0, 0, 0, 0, 0, 5_000.0, 0L, 0L, w);
    assertThat(s).isCloseTo(0.0, Offset.offset(1e-9));
  }

  @Test
  void rejectsNegativeWeights() {
    org.junit.jupiter.api.Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> ObjectiveWeights.builder().recallWeight(-0.1).build());
  }
}
