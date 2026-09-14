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

import java.util.Arrays;
import java.util.Random;

/** Fresh-JVM probe used by {@link F32ExecutionMatrixColdHotTest}. */
final class F32ExecutionMatrixColdHotProbe {

  private F32ExecutionMatrixColdHotProbe() {}

  public static void main(String[] args) {
    assertStable(32, 2_048);
    assertStable(2_048, 32);
  }

  private static void assertStable(int rows, int columns) {
    Random random = new Random(0xF32C_01D5L + rows * 31L + columns);
    float[] input = new float[columns];
    float[] weights = new float[Math.multiplyExact(rows, columns)];
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 4.0f - 2.0f;
    }
    for (int index = 0; index < weights.length; index++) {
      weights[index] = random.nextFloat() * 0.1f - 0.05f;
    }
    F32ExecutionMatrix matrix =
        F32ExecutionMatrix.copyOf(weights, rows, columns, (long) weights.length * Float.BYTES);
    float[] first = new float[rows];
    float[] hot = new float[rows];

    matrix.multiplyBatch(input, 1, first);
    for (int iteration = 0; iteration < 256; iteration++) {
      matrix.multiplyBatch(input, 1, hot);
    }

    if (Arrays.equals(first, hot)) {
      return;
    }
    for (int index = 0; index < first.length; index++) {
      if (Float.floatToRawIntBits(first[index]) != Float.floatToRawIntBits(hot[index])) {
        throw new AssertionError(
            "owned F32 matrix "
                + rows
                + "x"
                + columns
                + " cold/hot mismatch at index "
                + index
                + ": first="
                + first[index]
                + ", hot="
                + hot[index]);
      }
    }
  }
}
