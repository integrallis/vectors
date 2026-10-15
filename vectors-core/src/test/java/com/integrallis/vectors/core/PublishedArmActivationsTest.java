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

import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Exercises the ARM recipe directly on every host; makes no architecture performance claim. */
@Tag("unit")
class PublishedArmActivationsTest {
  private static final float CANARY = -713.25f;
  private static final float[] INPUTS = {
    0f,
    -0f,
    Float.MIN_VALUE,
    -Float.MIN_VALUE,
    Float.MIN_NORMAL,
    -Float.MIN_NORMAL,
    -105f,
    -104f,
    -100f,
    -88f,
    -80f,
    -20f,
    -1f,
    -.125f,
    .125f,
    1f,
    Math.nextDown(20f),
    20f,
    Math.nextUp(20f),
    80f,
    100f,
    104f,
    105f,
    Float.POSITIVE_INFINITY,
    Float.NEGATIVE_INFINITY,
    Float.NaN
  };

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 33, 65})
  void swiGluPreservesOffsetsInputsAndTheScalarRecipe(int length) {
    float[] gate = inputs(length, 3, 0);
    float[] up = inputs(length, 5, 11);
    float[] gateBefore = gate.clone();
    float[] upBefore = up.clone();
    float[] result = padded(length, 7);
    float[] expected = result.clone();
    for (int i = 0; i < length; i++) {
      float x = gate[i + 3];
      expected[i + 7] = x / (1f + ArmActivationReference.exp(-x)) * up[i + 5];
    }
    PublishedArmActivations.swiGlu(result, 7, gate, 3, up, 5, length);
    sameBits(result, expected);
    sameBits(gate, gateBefore);
    sameBits(up, upBefore);
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 3, 4, 5, 8, 9, 17, 33})
  void swiGluSupportsEachExactInPlaceOutput(int length) {
    for (int alias = 0; alias < 3; alias++) {
      float[] gate = inputs(length, 3, 0);
      float[] up = alias == 2 ? gate : inputs(length, 3, 9);
      float[] out = alias == 1 ? up : gate;
      float[] expected = out.clone();
      for (int i = 0; i < length; i++) {
        float x = gate[i + 3];
        expected[i + 3] = x / (1f + ArmActivationReference.exp(-x)) * up[i + 3];
      }
      PublishedArmActivations.swiGlu(out, 3, gate, 3, up, 3, length);
      sameBits(out, expected);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 33, 65})
  void gatesPreserveOffsetsAndSupportInPlaceSoftplus(int length) {
    for (boolean inPlace : new boolean[] {false, true}) {
      float[] beta = inputs(length, 3, 0);
      float[] alpha = inputs(length, 5, 7);
      float[] bias = padded(length, 7);
      float[] scale = padded(length, 9);
      for (int i = 0; i < length; i++) {
        bias[i + 7] = (i % 3 - 1) * .125f;
        scale[i + 9] = (i % 5 - 2) * .5f;
      }
      float[] alphaBefore = alpha.clone();
      float[] biasBefore = bias.clone();
      float[] scaleBefore = scale.clone();
      float[] out = inPlace ? alpha : padded(length, 5);
      float[] expectedOut = out.clone();
      float[] expectedBeta = beta.clone();
      for (int i = 0; i < length; i++) {
        expectedBeta[i + 3] = 1f / (1f + ArmActivationReference.exp(-beta[i + 3]));
        float value = alpha[i + 5] + bias[i + 7];
        float softplus =
            value > 20f
                ? value
                : ArmActivationReference.log(1f + ArmActivationReference.exp(value));
        expectedOut[i + 5] = scale[i + 9] * softplus;
      }
      PublishedArmActivations.sigmoidAndScaledSoftplus(
          beta, 3, alpha, 5, bias, 7, scale, 9, out, 5, length);
      sameBits(beta, expectedBeta);
      sameBits(out, expectedOut);
      if (!inPlace) sameBits(alpha, alphaBefore);
      sameBits(bias, biasBefore);
      sameBits(scale, scaleBefore);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 33, 65})
  void convolutionPreservesSuccessiveHistoryAndTheSelectedAccumulationOrder(int channels) {
    for (int kernel : new int[] {2, 4, 7}) {
      float[] history = padded((kernel - 1) * channels, 5);
      float[] weights = padded(kernel * channels, 7);
      for (int i = 0; i < (kernel - 1) * channels; i++) history[i + 5] = (i % 17 - 8) * .03125f;
      for (int i = 0; i < kernel * channels; i++) weights[i + 7] = (i % 13 - 6) * .0625f;
      float[] expectedHistory = history.clone();
      float[] originalWeights = weights.clone();
      for (int step = 0; step < 4; step++) {
        float[] values = padded(channels, 3);
        for (int channel = 0; channel < channels; channel++)
          values[channel + 3] = (channel % 11 - 5) * .3125f + step * .03125f;
        float[] expected = values.clone();
        referenceConvolution(expected, expectedHistory, weights, channels, kernel);
        PublishedArmActivations.causalDepthwiseConv1dSilu(
            values, 3, history, 5, weights, 7, channels, kernel);
        sameBits(values, expected);
        sameBits(history, expectedHistory);
        sameBits(weights, originalWeights);
      }
    }
  }

  @Test
  void convolutionHandlesNonfiniteAndSignedZeroInFullGroupsAndTails() {
    for (int channels : new int[] {3, 4, 5, 33, 65}) {
      float[] values = inputs(channels, 3, 0);
      float[] history = padded(channels, 5);
      float[] weights = padded(2 * channels, 7);
      Arrays.fill(history, 5, 5 + channels, 0f);
      Arrays.fill(weights, 7, 7 + 2 * channels, 1f);
      float[] expected = values.clone();
      float[] expectedHistory = history.clone();
      referenceConvolution(expected, expectedHistory, weights, channels, 2);
      PublishedArmActivations.causalDepthwiseConv1dSilu(
          values, 3, history, 5, weights, 7, channels, 2);
      sameBits(values, expected);
      sameBits(history, expectedHistory);
    }
  }

  @Test
  void vectorAndScalarTailSwiGluKeepTheSameOwnedSnapshotAfterWarmup() {
    float[] gate = inputs(65, 3, 0);
    float[] up = inputs(65, 5, 7);
    float[] first = padded(65, 7);
    PublishedArmActivations.swiGlu(first, 7, gate, 3, up, 5, 65);
    float[] warmed = padded(65, 7);
    for (int repeat = 0; repeat < 1024; repeat++)
      PublishedArmActivations.swiGlu(warmed, 7, gate, 3, up, 5, 65);
    sameBits(warmed, first);
    for (int i = 0; i < 65; i++)
      PublishedArmActivations.swiGlu(warmed, 7 + i, gate, 3 + i, up, 5 + i, 1);
    sameBits(warmed, first);
  }

  @Test
  void invalidSourceRangesAreRejectedForVectorAndScalarAccesses() {
    float[] data = new float[8];
    assertThatThrownBy(() -> PublishedArmActivations.swiGlu(data, 0, data, -1, data, 0, 1))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> PublishedArmActivations.swiGlu(data, 0, data, 6, data, 0, 4))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(
            () ->
                PublishedArmActivations.sigmoidAndScaledSoftplus(
                    data, -1, data, 0, data, 0, data, 0, data, 0, 1))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(
            () ->
                PublishedArmActivations.causalDepthwiseConv1dSilu(data, -1, data, 0, data, 0, 1, 2))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  private static void referenceConvolution(
      float[] values, float[] history, float[] weights, int channels, int kernel) {
    int vectorEnd = channels - channels % 4;
    for (int channel = 0; channel < channels; channel++) {
      float input = values[channel + 3];
      float sum = input * weights[7 + (kernel - 1) * channels + channel];
      for (int tap = 0; tap < kernel - 1; tap++) {
        int at = 5 + tap * channels + channel;
        float previous = history[at];
        float weight = weights[7 + tap * channels + channel];
        sum =
            channel < vectorEnd && PanamaConstants.HAS_FAST_VECTOR_FMA
                ? Math.fma(previous, weight, sum)
                : sum + previous * weight;
        if (tap > 0) history[at - channels] = previous;
      }
      history[5 + (kernel - 2) * channels + channel] = input;
      values[channel + 3] = sum / (1f + ArmActivationReference.exp(-sum));
    }
  }

  private static float[] padded(int length, int offset) {
    float[] result = new float[offset + length + 11];
    Arrays.fill(result, CANARY);
    return result;
  }

  private static float[] inputs(int length, int offset, int rotation) {
    float[] result = padded(length, offset);
    for (int i = 0; i < length; i++) result[i + offset] = INPUTS[(i + rotation) % INPUTS.length];
    return result;
  }

  private static void sameBits(float[] actual, float[] expected) {
    assertThat(actual).hasSameSizeAs(expected);
    for (int i = 0; i < actual.length; i++) {
      assertThat(Float.floatToIntBits(actual[i]))
          .as("float bits at %s", i)
          .isEqualTo(Float.floatToIntBits(expected[i]));
    }
  }
}
