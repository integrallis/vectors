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

import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ConvolutionSiluContractTest {
  @Test
  void disjointBuffersOffsetsTailsAndHistoryMatchTheFloatContract() {
    for (int channels : new int[] {1, 9, 17, 33}) {
      for (int kernel : new int[] {2, 4, 7}) check(channels, kernel, 0);
    }
  }

  @Test
  void sharedArraysRetainPublishedVectorGroupReadAndUpdateOrder() {
    for (int channels : new int[] {9, 17, 31}) {
      // Includes overlapping values/history, weights, and shared but disjoint subranges.
      for (int alias = 1; alias <= 5; alias++) check(channels, 4, alias);
    }
  }

  @Test
  void specialValuesAndOverflowFollowThePublicFloatAndHistoryContract() {
    float[] edges = {
      0.0f,
      -0.0f,
      Float.MIN_VALUE,
      -Float.MIN_VALUE,
      Float.MIN_NORMAL,
      -Float.MIN_NORMAL,
      Float.MAX_VALUE,
      -Float.MAX_VALUE,
      Float.POSITIVE_INFINITY,
      Float.NEGATIVE_INFINITY,
      Float.NaN,
      100f,
      -100f,
      104f,
      -104f,
      Float.MAX_VALUE,
      -Float.MAX_VALUE
    };
    int channels = edges.length, offset = 3, kernel = 4;
    float[] values = new float[channels + 2 * offset];
    System.arraycopy(edges, 0, values, offset, channels);
    float[] history = new float[3 * channels + 2 * offset];
    float[] weights = new float[4 * channels + 2 * offset];
    Arrays.fill(weights, offset + 3 * channels, offset + 4 * channels, 1f);
    weights[offset + 4 * channels - 1] = 2f;
    weights[offset + 4 * channels - 2] = 2f;
    float[] expectedValues = values.clone(), expectedHistory = history.clone();
    reference(expectedValues, offset, expectedHistory, offset, weights, offset, channels, kernel);
    VectorUtil.causalDepthwiseConv1dSilu(
        values, offset, history, offset, weights, offset, channels, kernel);
    for (int i = 0; i < values.length; i++) {
      assertThat(Float.floatToIntBits(values[i]))
          .as("public convolution cell %s", i)
          .isEqualTo(Float.floatToIntBits(expectedValues[i]));
    }
    for (int i = 0; i < history.length; i++) {
      assertThat(Float.floatToIntBits(history[i]))
          .as("public history cell %s", i)
          .isEqualTo(Float.floatToIntBits(expectedHistory[i]));
    }
  }

  private static void check(int channels, int kernel, int alias) {
    float[][] actual = fixture(channels, alias), expected = fixture(channels, alias);
    int valuesOffset = 1, historyOffset = alias == 5 ? 64 : 2;
    int weightsOffset = alias == 5 ? 256 : 3;
    VectorUtil.causalDepthwiseConv1dSilu(
        actual[0],
        valuesOffset,
        actual[1],
        historyOffset,
        actual[2],
        weightsOffset,
        channels,
        kernel);
    reference(
        expected[0],
        valuesOffset,
        expected[1],
        historyOffset,
        expected[2],
        weightsOffset,
        channels,
        kernel);
    for (int array = 0; array < actual.length; array++) {
      for (int i = 0; i < actual[array].length; i++) {
        assertThat(Float.floatToIntBits(actual[array][i]))
            .as(
                "channels=%s kernel=%s alias=%s array=%s cell=%s",
                channels, kernel, alias, array, i)
            .isEqualTo(Float.floatToIntBits(expected[array][i]));
      }
    }
  }

  private static float[][] fixture(int channels, int alias) {
    float[] values = new float[512], history = new float[512], weights = new float[512];
    SplittableRandom random = new SplittableRandom(9182 + channels);
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) random.nextDouble(-1, 1);
      history[i] = (float) random.nextDouble(-1, 1);
      weights[i] = (float) random.nextDouble(-1, 1);
    }
    if (alias == 1) history = values;
    if (alias == 2) weights = values;
    if (alias == 3) weights = history;
    if (alias >= 4) {
      history = values;
      weights = values;
    }
    return new float[][] {values, history, weights};
  }

  // Scalar replay of the published full-width group contract: snapshot each vector load before
  // the corresponding stores. This matters when history and values share storage.
  private static void reference(
      float[] values,
      int vo,
      float[] history,
      int ho,
      float[] weights,
      int wo,
      int channels,
      int kernel) {
    int width =
        VectorizationProvider.isPanamaEnabled()
            ? PanamaVectorUtilSupport.FLOAT_SPECIES.length()
            : 1;
    int limit = VectorizationProvider.isPanamaEnabled() ? channels - channels % width : 0;
    for (int channel = 0; channel < channels; ) {
      int count = channel < limit ? width : 1;
      float[] input = Arrays.copyOfRange(values, vo + channel, vo + channel + count);
      float[] sum = new float[count];
      for (int lane = 0; lane < count; lane++) {
        sum[lane] = input[lane] * weights[wo + (kernel - 1) * channels + channel + lane];
      }
      for (int tap = 0; tap < kernel - 1; tap++) {
        int hi = ho + tap * channels + channel;
        float[] previous = Arrays.copyOfRange(history, hi, hi + count);
        for (int lane = 0; lane < count; lane++) {
          float weight = weights[wo + tap * channels + channel + lane];
          sum[lane] =
              channel < limit && PanamaConstants.HAS_FAST_VECTOR_FMA
                  ? Math.fma(previous[lane], weight, sum[lane])
                  : previous[lane] * weight + sum[lane];
        }
        if (tap > 0) System.arraycopy(previous, 0, history, hi - channels, count);
      }
      System.arraycopy(input, 0, history, ho + (kernel - 2) * channels + channel, count);
      // This oracle tests read/write ordering, not the independently tested activation error.
      // Using the actual activation prevents approximate output fed through aliases from changing
      // later expected history values merely because the scalar reference rounds differently.
      if (ArmActivationReference.applies()) {
        // The ARM recipe uses the same explicit exponential for full groups and scalar tails.
        // Replay it independently; calling the previous activation implementation here would
        // test a different numerical contract and then contaminate subsequent aliased history.
        for (int lane = 0; lane < count; lane++) {
          sum[lane] = sum[lane] / (1.0f + ArmActivationReference.exp(-sum[lane]));
        }
      } else if (channel < limit) {
        ReproducibleConvolutionSilu.apply(sum, 0, count);
      } else {
        sum[0] = sum[0] / (1.0f + (float) StrictMath.exp(-sum[0]));
      }
      System.arraycopy(sum, 0, values, vo + channel, count);
      channel += count;
    }
  }
}
