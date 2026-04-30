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
package com.integrallis.vectors.hybrid;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class WeightedFusionTest {

  @Test
  void equalWeightsMinMaxNormalization() {
    var fusion = new WeightedFusion(1.0f, 1.0f);
    // List 1: a=0.8, b=0.2 → normalized: a=1.0, b=0.0
    // List 2: a=0.5, c=1.0 → normalized: a=0.0, c=1.0
    List<ScoredId> list1 = List.of(new ScoredId("a", 0.8f), new ScoredId("b", 0.2f));
    List<ScoredId> list2 = List.of(new ScoredId("c", 1.0f), new ScoredId("a", 0.5f));

    List<ScoredId> fused = fusion.fuse(List.of(list1, list2), 3);

    assertThat(fused).hasSize(3);
    // a: 1.0*1.0 + 0.0*1.0 = 1.0; c: 1.0*1.0 = 1.0 (tie-break by id)
    assertThat(fused.get(0).id()).isEqualTo("a");
    assertThat(fused.get(0).score()).isCloseTo(1.0f, Offset.offset(1e-6f));
  }

  @Test
  void unequalWeightsBoostRetriever() {
    var fusion = new WeightedFusion(2.0f, 0.5f);
    // List 1 (weight=2.0): a=1.0 → normalized a=1.0 (singleton, so 1.0)
    // List 2 (weight=0.5): b=1.0 → normalized b=1.0
    List<ScoredId> list1 = List.of(new ScoredId("a", 1.0f));
    List<ScoredId> list2 = List.of(new ScoredId("b", 1.0f));

    List<ScoredId> fused = fusion.fuse(List.of(list1, list2), 2);

    assertThat(fused.get(0).id()).isEqualTo("a");
    assertThat(fused.get(0).score()).isCloseTo(2.0f, Offset.offset(1e-6f));
    assertThat(fused.get(1).id()).isEqualTo("b");
    assertThat(fused.get(1).score()).isCloseTo(0.5f, Offset.offset(1e-6f));
  }

  @Test
  void singleSourceDegenerateCase() {
    var fusion = new WeightedFusion(1.0f);
    List<ScoredId> list = List.of(new ScoredId("a", 0.9f), new ScoredId("b", 0.1f));

    List<ScoredId> fused = fusion.fuse(List.of(list), 2);

    assertThat(fused).hasSize(2);
    assertThat(fused.get(0).id()).isEqualTo("a");
    // a normalized = (0.9 - 0.1) / (0.9 - 0.1) = 1.0
    assertThat(fused.get(0).score()).isCloseTo(1.0f, Offset.offset(1e-6f));
    assertThat(fused.get(1).id()).isEqualTo("b");
    assertThat(fused.get(1).score()).isCloseTo(0.0f, Offset.offset(1e-6f));
  }

  @Test
  void mismatchedListCountThrows() {
    var fusion = new WeightedFusion(1.0f, 2.0f);
    assertThatThrownBy(() -> fusion.fuse(List.of(List.of()), 5))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void emptyWeightsThrows() {
    assertThatThrownBy(() -> new WeightedFusion(new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void nonPositiveWeightThrows() {
    assertThatThrownBy(() -> new WeightedFusion(1.0f, 0f))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new WeightedFusion(-1.0f))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
