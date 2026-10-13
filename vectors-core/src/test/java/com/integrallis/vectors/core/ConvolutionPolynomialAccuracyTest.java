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

import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ConvolutionPolynomialAccuracyTest {
  @Test
  void publishedAccuracyAndExactColdHotStability() {
    float[] initial = new float[4096];
    SplittableRandom random = new SplittableRandom(981371);
    for (int i = 0; i < initial.length; i++) initial[i] = (float) random.nextDouble(-32, 32);
    initial[0] = 0f;
    initial[1] = -0f;
    initial[2] = Float.MIN_VALUE;
    initial[3] = -Float.MIN_VALUE;
    float[] cold = initial.clone();
    ReproducibleConvolutionSilu.apply(cold, 0, cold.length);
    for (int i = 0; i < cold.length; i++) {
      float expected = initial[i] / (1f + (float) Math.exp(-initial[i]));
      assertThat(Math.abs((double) cold[i] - expected)).isLessThanOrEqualTo(2e-6);
      if (expected == 0f)
        assertThat(Float.floatToRawIntBits(cold[i])).isEqualTo(Float.floatToRawIntBits(expected));
    }
    float[] hot = new float[initial.length];
    for (int repeat = 0; repeat < 2000; repeat++) {
      System.arraycopy(initial, 0, hot, 0, initial.length);
      ReproducibleConvolutionSilu.apply(hot, 0, hot.length);
      assertThat(hot).containsExactly(cold);
    }
  }

  @Test
  void ordinaryIntervalBoundariesAndRecordedErrorExtremaMeetPublishedAccuracy() {
    float[] input = {
      -32f,
      Math.nextUp(-32f),
      Math.nextDown(32f),
      32f,
      Float.intBitsToFloat(0x4101bcc2),
      Float.intBitsToFloat(0xc1ff30d6)
    };
    float[] actual = input.clone();
    ReproducibleConvolutionSilu.apply(actual, 0, actual.length);
    for (int i = 0; i < input.length; i++) {
      float expected = input[i] / (1f + (float) Math.exp(-input[i]));
      assertThat(Math.abs((double) actual[i] - expected)).isLessThanOrEqualTo(2e-6);
    }
    for (float outside : new float[] {Math.nextDown(-32f), Math.nextUp(32f)}) {
      float[] mixed = {outside, Float.intBitsToFloat(0x4001b249), -0f};
      float[] expected = mixed.clone();
      for (int i = 0; i < expected.length; i++)
        expected[i] = expected[i] / (1f + (float) StrictMath.exp(-expected[i]));
      ReproducibleConvolutionSilu.apply(mixed, 0, mixed.length);
      for (int i = 0; i < mixed.length; i++)
        assertThat(Float.floatToRawIntBits(mixed[i]))
            .isEqualTo(Float.floatToRawIntBits(expected[i]));
    }
  }

  @Test
  void exceptionalBlockKeepsOrdinaryAndNonfiniteLanesReproducible() {
    float[] input = {
      Float.POSITIVE_INFINITY,
      Float.NEGATIVE_INFINITY,
      Float.NaN,
      0f,
      -0f,
      Float.MIN_VALUE,
      -Float.MIN_VALUE,
      Float.MAX_VALUE,
      -Float.MAX_VALUE,
      Float.intBitsToFloat(0xb3000000),
      Float.intBitsToFloat(0x4001b249),
      Float.intBitsToFloat(0x4288942b)
    };
    float[] actual = input.clone();
    ReproducibleConvolutionSilu.apply(actual, 0, actual.length);
    for (int i = 0; i < input.length; i++) {
      float expected = input[i] / (1f + (float) StrictMath.exp(-input[i]));
      assertThat(Float.floatToIntBits(actual[i])).isEqualTo(Float.floatToIntBits(expected));
    }
  }
}
