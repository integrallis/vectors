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
package com.integrallis.vectors.optimizer.space;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ParamSpecTest {

  @Test
  void categoricalCardinalityMatchesValues() {
    var c = new ParamSpec.Categorical("metric", List.of("COSINE", "EUCLIDEAN", "DOT"));
    assertThat(c.cardinality()).hasValue(3L);
    assertThat(c.values()).containsExactly("COSINE", "EUCLIDEAN", "DOT");
  }

  @Test
  void intRangeRejectsInvertedBounds() {
    assertThatThrownBy(() -> new ParamSpec.IntRange("m", 64, 16, false))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("min");
  }

  @Test
  void intRangeCardinalityIsInclusive() {
    var r = new ParamSpec.IntRange("efConstruction", 100, 200, false);
    assertThat(r.cardinality()).hasValue(101L);
  }

  @Test
  void doubleRangeLogScaleRequiresPositiveMin() {
    assertThatThrownBy(() -> new ParamSpec.DoubleRange("alpha", 0.0, 1.0, true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("logScale");
  }

  @Test
  void doubleRangeIsContinuous() {
    var r = new ParamSpec.DoubleRange("alpha", 0.1, 1.0, true);
    assertThat(r.cardinality()).isEmpty();
  }

  @Test
  void fixedSpecsHaveCardinalityOne() {
    assertThat(new ParamSpec.FixedString("k", "FLAT").cardinality()).hasValue(1L);
    assertThat(new ParamSpec.FixedInt("dim", 768).cardinality()).hasValue(1L);
    assertThat(new ParamSpec.FixedDouble("threshold", 0.5).cardinality()).hasValue(1L);
  }

  @Test
  void discreteRejectsEmptyValues() {
    assertThatThrownBy(() -> new ParamSpec.Discrete<Integer>("subspaces", List.of()))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
