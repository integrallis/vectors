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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Direct fallback tests remain meaningful when the public host route uses another backend. */
class FallbackSwiGluContractTest {
  private static final float CANARY = -12345.25f;
  private static final float[] EDGES = {
    0f,
    -0f,
    Float.MIN_VALUE,
    -Float.MIN_VALUE,
    Float.MIN_NORMAL,
    -Float.MIN_NORMAL,
    80f,
    -80f,
    Math.nextUp(80f),
    Math.nextDown(-80f),
    Float.MAX_VALUE,
    -Float.MAX_VALUE,
    Float.POSITIVE_INFINITY,
    Float.NEGATIVE_INFINITY,
    Float.NaN
  };

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 33, 65})
  void ordinaryAndExceptionalInputsKeepAccuracyOffsetsAndChunkIdentity(int length) {
    for (boolean exceptional : new boolean[] {false, true}) {
      float[] gate = new float[length + 15], up = new float[length + 15];
      Random random = new Random(9043 + length);
      for (int i = 0; i < gate.length; i++) {
        gate[i] = exceptional ? EDGES[i % EDGES.length] : random.nextFloat() * 20f - 10f;
        up[i] = random.nextFloat() * 20f - 10f;
      }
      float[] originalGate = gate.clone(), originalUp = up.clone();
      float[] whole = new float[length + 15],
          chunks = new float[whole.length],
          delegated = new float[whole.length];
      Arrays.fill(whole, CANARY);
      Arrays.fill(chunks, CANARY);
      Arrays.fill(delegated, CANARY);
      StableFloatSwiGlu.apply(whole, 7, gate, 7, up, 7, length);
      for (int i = 0; i < length; i++)
        StableFloatSwiGlu.apply(chunks, 7 + i, gate, 7 + i, up, 7 + i, 1);
      ReproducibleActivationExp.swiGlu(delegated, 7, gate, 7, up, 7, length);
      for (int i = 0; i < whole.length; i++) {
        bits(whole[i], chunks[i], "chunk " + i);
        bits(whole[i], delegated[i], "delegated " + i);
        if (i < 7 || i >= length + 7) bits(CANARY, whole[i], "canary " + i);
        else accuracy(gate[i], up[i], whole[i]);
        bits(originalGate[i], gate[i], "gate preserved " + i);
        bits(originalUp[i], up[i], "up preserved " + i);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 3, 8, 17, 33})
  void inPlaceAndBackwardOverlapMatchOwnedSnapshot(int length) {
    for (int alias = 0; alias < 3; alias++) {
      for (int outputAt : new int[] {0, 2}) {
        float[] gate = new float[length + 6], up = new float[length + 6];
        for (int i = 0; i < gate.length; i++) {
          gate[i] = (i - 5) * .125f;
          up[i] = (i + 1) * -.25f;
        }
        if (alias == 2) up = gate;
        float[] originalGate = gate.clone(), originalUp = up.clone();
        float[] out = alias == 1 ? up : gate;
        float[] originalOut = out.clone(), expected = new float[length];
        StableFloatSwiGlu.apply(expected, 0, originalGate, 2, originalUp, 2, length);
        ReproducibleActivationExp.swiGlu(out, outputAt, gate, 2, up, 2, length);
        for (int i = 0; i < out.length; i++) {
          bits(
              i >= outputAt && i < outputAt + length ? expected[i - outputAt] : originalOut[i],
              out[i],
              "alias " + alias + " at " + i);
        }
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2})
  void forwardOverlapRetainsFirstGroupAccuracyAndColdHotHistory(int alias) {
    int length = PanamaVectorUtilSupport.FLOAT_SPECIES.length() * 3 + 1;
    int firstGroup = PanamaVectorUtilSupport.FLOAT_SPECIES.length();
    float[] gate = new float[length + 3], up = new float[length + 3];
    float[] values = {-110f, -12.5f, -0f, 0f, 80f, 100f, Float.NaN};
    for (int i = 0; i < gate.length; i++) {
      gate[i] = values[i % values.length];
      up[i] = .25f * (i + 1);
    }
    if (alias == 2) up = gate;
    float[] originalGate = gate.clone(), originalUp = up.clone();
    float[] output = alias == 1 ? up : gate;
    float[] originalOut = output.clone();
    ReproducibleActivationExp.swiGlu(output, 1, gate, 0, up, 0, length);
    float[] cold = output.clone();
    for (int i = 0; i < firstGroup; i++) {
      float expected =
          originalGate[i] / (1f + (float) StrictMath.exp(-originalGate[i])) * originalUp[i];
      bits(expected, cold[i + 1], "first independently owned read group " + i);
    }
    bits(originalOut[0], cold[0], "leading canary");
    for (int i = length + 1; i < output.length; i++)
      bits(originalOut[i], cold[i], "trailing canary");
    for (int repeat = 0; repeat < 128; repeat++) {
      System.arraycopy(originalGate, 0, gate, 0, gate.length);
      if (up != gate) System.arraycopy(originalUp, 0, up, 0, up.length);
      ReproducibleActivationExp.swiGlu(output, 1, gate, 0, up, 0, length);
      for (int i = 0; i < output.length; i++)
        bits(cold[i], output[i], "repeat " + repeat + " lane " + i);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2})
  void forwardOverlapPreservesCompletePublishedReadGroups(int alias) {
    int lanes = PanamaVectorUtilSupport.FLOAT_SPECIES.length();
    for (int length : new int[] {1, lanes - 1, lanes, lanes + 1, 3 * lanes + 1, 67}) {
      for (int shift = 1; shift <= lanes + 2 && shift < length; shift++) {
        float[] gate = new float[length + shift + 5], up = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
          gate[i] = .25f * (i % 13 - 5);
          up[i] = .5f + (i % 3) * .125f;
        }
        if (alias == 2) up = gate;
        float[] expectedGate = gate.clone(), expectedUp = alias == 2 ? expectedGate : up.clone();
        float[] output = alias == 1 ? up : gate;
        float[] expected = alias == 1 ? expectedUp : expectedGate;
        int inputOffset = 2, outputOffset = inputOffset + shift;
        int i = 0, limit = length - length % lanes;
        for (; i < limit; i += lanes) {
          float[] groupGate =
              Arrays.copyOfRange(expectedGate, inputOffset + i, inputOffset + i + lanes);
          float[] groupUp =
              Arrays.copyOfRange(expectedUp, inputOffset + i, inputOffset + i + lanes);
          for (int j = 0; j < lanes; j++) {
            float g = groupGate[j];
            expected[outputOffset + i + j] = g / (1f + (float) StrictMath.exp(-g)) * groupUp[j];
          }
        }
        for (; i < length; i++) {
          float g = expectedGate[inputOffset + i];
          expected[outputOffset + i] =
              g / (1f + (float) StrictMath.exp(-g)) * expectedUp[inputOffset + i];
        }
        ReproducibleActivationExp.swiGlu(
            output, outputOffset, gate, inputOffset, up, inputOffset, length);
        for (i = 0; i < output.length; i++) {
          bits(
              expected[i],
              output[i],
              "alias=" + alias + " shift=" + shift + " length=" + length + " index=" + i);
        }
      }
    }
  }

  private static void accuracy(float gate, float up, float actual) {
    float expected = gate / (1f + (float) StrictMath.exp(-gate)) * up;
    if (!Float.isFinite(expected) || expected == 0f) bits(expected, actual, "special value");
    else
      assertTrue(
          Math.abs((double) actual - expected) <= 2e-6 * Math.abs(up) + Math.ulp(expected),
          "published SiLU accuracy gate="
              + gate
              + " up="
              + up
              + " expected="
              + expected
              + " actual="
              + actual);
  }

  private static void bits(float expected, float actual, String context) {
    assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual), context);
  }
}
