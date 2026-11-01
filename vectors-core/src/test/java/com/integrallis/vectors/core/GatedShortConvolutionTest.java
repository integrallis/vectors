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

import static org.assertj.core.api.Assertions.*;

import java.util.*;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

class GatedShortConvolutionTest {
  @Test
  void scalarAndAllVectorWidthsPreserveTapOrderAndChunkHistory() {
    Random random = new Random(631);
    for (int channels : new int[] {1, 7, 8, 17, 64})
      for (int width : new int[] {3, 7}) {
        int tokens = 19;
        float[] projected = values(random, tokens * 3 * channels),
            weights = values(random, width * channels);
        float[] expected = new float[tokens * channels],
            history = new float[(width - 1) * channels];
        GatedShortConvolution.scalar(
            expected, 0, projected, 0, tokens, channels, weights, history, width);
        for (var species :
            List.of(FloatVector.SPECIES_128, FloatVector.SPECIES_256, FloatVector.SPECIES_512)) {
          float[] actual = new float[expected.length], state = new float[history.length];
          GatedShortConvolution.vector(
              actual, 0, projected, 0, 5, channels, weights, state, width, species);
          GatedShortConvolution.vector(
              actual,
              5 * channels,
              projected,
              15 * channels,
              tokens - 5,
              channels,
              weights,
              state,
              width,
              species);
          assertThat(actual).containsExactly(expected);
          assertThat(state).containsExactly(history);
        }
      }
  }

  @Test
  void appliesBothGatesWithoutSiluAndOldestTapFirst() {
    float[] result = new float[2], history = {2, 3};
    // B,C,X per token. First output=5*(1*2+10*3+100*(2*4))=4160.
    VectorUtil.gatedShortConv1d(
        result, 0, new float[] {2, 5, 4, 3, -2, 2}, 0, 2, 1, new float[] {1, 10, 100}, history, 3);
    assertThat(result).containsExactly(4160f, -1366f);
    assertThat(history).containsExactly(8f, 6f);
  }

  @Test
  void validatesBuffersBeforeUpdatingHistory() {
    float[] history = {1, 2};
    assertThatThrownBy(
            () ->
                VectorUtil.gatedShortConv1d(
                    new float[1], 0, new float[2], 0, 1, 1, new float[3], history, 3))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(history).containsExactly(1f, 2f);
    assertThatThrownBy(
            () ->
                VectorUtil.gatedShortConv1d(
                    history, 0, new float[3], 0, 1, 1, new float[3], history, 3))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static float[] values(Random random, int count) {
    float[] x = new float[count];
    for (int i = 0; i < count; i++) x[i] = random.nextFloat() * 4 - 2;
    return x;
  }
}
