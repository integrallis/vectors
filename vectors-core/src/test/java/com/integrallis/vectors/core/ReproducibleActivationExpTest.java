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
import org.junit.jupiter.api.Test;

class ReproducibleActivationExpTest {
  @Test
  void publicSwiGluSpecialValuesAndSampledFloatBitsMatchStrictMathExactly() {
    var species = PanamaVectorUtilSupport.FLOAT_SPECIES;
    float[] input = new float[species.length()];
    float[] output = new float[species.length()];
    float[] up = new float[species.length()];
    java.util.Arrays.fill(up, 1.0f);
    float[] edges = {
      0f,
      -0f,
      Float.MIN_VALUE,
      -Float.MIN_VALUE,
      Float.MIN_NORMAL,
      -Float.MIN_NORMAL,
      Float.MAX_VALUE,
      -Float.MAX_VALUE,
      Float.POSITIVE_INFINITY,
      Float.NEGATIVE_INFINITY,
      Float.NaN,
      89f,
      Math.nextDown(89f),
      -104f,
      Math.nextUp(-104f),
      0x1p-28f,
      -0x1p-28f
    };
    SplittableRandom random = new SplittableRandom(417326);
    for (int sample = 0; sample < 100_000; sample += input.length) {
      for (int lane = 0; lane < input.length; lane++) {
        int index = sample + lane;
        input[lane] = index < edges.length ? edges[index] : Float.intBitsToFloat(random.nextInt());
      }
      VectorUtil.swiGlu(output, 0, input, 0, up, 0, input.length);
      for (int lane = 0; lane < input.length; lane++) {
        assertThat(Float.floatToIntBits(output[lane]))
            .as("float input bits %08x", Float.floatToRawIntBits(input[lane]))
            .isEqualTo(
                Float.floatToIntBits(input[lane] / (1.0f + (float) StrictMath.exp(-input[lane]))));
      }
    }
  }

  @Test
  void publicSwiGluPreservesStrictFloatIntermediatesAndExactAlias() {
    SplittableRandom random = new SplittableRandom(816371);
    for (int length : new int[] {1, 2, 3, 7, 8, 9, 31, 32, 33, 259, 4096}) {
      float[] gate = new float[length + 6], up = new float[length + 6];
      float[] expected = new float[length];
      for (int i = 0; i < gate.length; i++) {
        gate[i] = (float) random.nextDouble(-100, 100);
        up[i] = (float) random.nextDouble(-4, 4);
      }
      for (int i = 0; i < length; i++) {
        float value = gate[i + 3];
        expected[i] = value / (1.0f + (float) StrictMath.exp(-value)) * up[i + 3];
      }
      float guard = up[0];
      VectorUtil.swiGlu(up, 3, gate, 3, up, 3, length);
      assertThat(up[0]).isEqualTo(guard);
      for (int i = 0; i < length; i++)
        assertThat(Float.floatToRawIntBits(up[i + 3]))
            .isEqualTo(Float.floatToRawIntBits(expected[i]));
    }
  }
}
