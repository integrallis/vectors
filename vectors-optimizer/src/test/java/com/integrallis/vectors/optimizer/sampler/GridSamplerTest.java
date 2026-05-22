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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GridSamplerTest {

  @Test
  void enumeratesCartesianProductInOrder() {
    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.Categorical("metric", List.of("COSINE", "DOT")),
                new ParamSpec.IntRange("m", 8, 10, false)));
    GridSampler g = new GridSampler(space);

    assertThat(g.total()).isEqualTo(2L * 3L);
    List<Trial> trials = new ArrayList<>();
    for (int i = 0; i < 6; i++) trials.add(g.next(List.of()));

    // Strides: axis 0 stride 1, axis 1 stride 2 (cardinality of axis 0).
    // Iteration: axis 0 cycles fastest.
    assertThat(trials.get(0).params()).containsEntry("metric", "COSINE").containsEntry("m", 8);
    assertThat(trials.get(1).params()).containsEntry("metric", "DOT").containsEntry("m", 8);
    assertThat(trials.get(2).params()).containsEntry("metric", "COSINE").containsEntry("m", 9);
    assertThat(trials.get(5).params()).containsEntry("metric", "DOT").containsEntry("m", 10);
  }

  @Test
  void throwsWhenExhausted() {
    SearchSpace space = new SearchSpace(List.of(new ParamSpec.Categorical("x", List.of("a", "b"))));
    GridSampler g = new GridSampler(space);
    g.next(List.of());
    g.next(List.of());
    assertThatThrownBy(() -> g.next(List.of()))
        .isInstanceOf(NoMoreTrialsException.class)
        .hasMessageContaining("exhausted");
  }

  @Test
  void rejectsContinuousAxis() {
    SearchSpace space =
        new SearchSpace(List.of(new ParamSpec.DoubleRange("alpha", 0.5, 1.5, false)));
    assertThatThrownBy(() -> new GridSampler(space))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("continuous")
        .hasMessageContaining("Discrete");
  }
}
