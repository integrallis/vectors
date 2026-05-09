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
class SearchSpaceTest {

  @Test
  void byNameReturnsAxis() {
    var m = new ParamSpec.Categorical("metric", List.of("COSINE", "DOT"));
    var k = new ParamSpec.IntRange("efConstruction", 50, 200, false);
    var space = new SearchSpace(List.of(m, k));

    assertThat(space.byName("metric")).hasValue(m);
    assertThat(space.byName("efConstruction")).hasValue(k);
  }

  @Test
  void byNameReturnsEmptyForUnknown() {
    var space = new SearchSpace(List.of(new ParamSpec.IntRange("m", 8, 64, false)));
    assertThat(space.byName("nonexistent")).isEmpty();
  }

  @Test
  void cardinalityIsInfiniteWhenAnyContinuous() {
    var space =
        new SearchSpace(
            List.of(
                new ParamSpec.IntRange("m", 8, 64, false),
                new ParamSpec.DoubleRange("alpha", 0.5, 1.5, false)));
    assertThat(space.cardinalityOrInfinite()).isEmpty();
  }

  @Test
  void cardinalityIsProductOfDiscreteAxes() {
    var space =
        new SearchSpace(
            List.of(
                new ParamSpec.Categorical("metric", List.of("COSINE", "DOT")),
                new ParamSpec.IntRange("m", 8, 16, false)));
    assertThat(space.cardinalityOrInfinite()).hasValue(2L * 9L);
  }

  @Test
  void duplicateAxisNameRejected() {
    assertThatThrownBy(
            () ->
                new SearchSpace(
                    List.of(
                        new ParamSpec.IntRange("m", 8, 16, false),
                        new ParamSpec.IntRange("m", 32, 64, false))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Duplicate");
  }
}
