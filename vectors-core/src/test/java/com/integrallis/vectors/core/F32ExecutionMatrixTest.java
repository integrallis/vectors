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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class F32ExecutionMatrixTest {

  @Test
  void ownsAndMultipliesARowMajorExecutionCopyForBatchAndRowTails() {
    float[] weights = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.5f, 4.0f,
      0.0f, -2.0f, 1.0f
    };
    F32ExecutionMatrix matrix = F32ExecutionMatrix.copyOf(weights, 3, 3, 18L);
    weights[0] = 99.0f;
    float[] output = new float[6];

    matrix.multiplyBatch(new float[] {1, 2, 3, -2, 1, 0.5f}, 2, output);

    assertThat(output).containsExactly(14.0f, 12.0f, -1.0f, 1.5f, 4.5f, -1.5f);
    assertThat(matrix.rows()).isEqualTo(3);
    assertThat(matrix.columns()).isEqualTo(3);
    assertThat(matrix.serializedByteCount()).isEqualTo(18L);
    assertThat(matrix.expandedByteCount()).isEqualTo(36L);
    assertThat(matrix.expansionRatio()).isEqualTo(2.0);
  }

  @Test
  void rejectsInvalidGeometryAndStorage() {
    assertThatThrownBy(() -> F32ExecutionMatrix.copyOf(null, 1, 1, 4L))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> F32ExecutionMatrix.copyOf(new float[2], 1, 1, 4L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rowMajorWeights");
    assertThatThrownBy(() -> F32ExecutionMatrix.copyOf(new float[1], 0, 1, 4L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
    assertThatThrownBy(() -> F32ExecutionMatrix.copyOf(new float[1], 1, 1, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("serializedByteCount");
  }
}
