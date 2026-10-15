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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/** Fresh-JVM public API probe for BF16 projections across compilation and batching. */
final class BFloat16MatrixColdHotProbe {
  private BFloat16MatrixColdHotProbe() {}

  public static void main(String[] args) {
    System.out.println(
        "BF16_ORDINARY_REDUCTION=" + (PanamaConstants.HAS_SVE ? "inline-sve" : "direct-helper"));
    int rows = Integer.parseInt(args[0]);
    int columns = Integer.parseInt(args[1]);
    Random random = new Random(0xBF16C01DL + rows * 31L + columns);
    float[] input = new float[columns];
    byte[] packed = new byte[rows * columns * Short.BYTES];
    ByteBuffer weights = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 4.0f - 2.0f;
    for (int i = 0; i < rows * columns; i++)
      weights.putShort((short) (Float.floatToRawIntBits(random.nextFloat() * 0.1f - 0.05f) >>> 16));
    try (Arena arena = Arena.ofShared()) {
      MemorySegment data = arena.allocate(packed.length, Long.BYTES);
      data.copyFrom(MemorySegment.ofArray(packed));
      BFloat16Matrix matrix = BFloat16Matrix.of(data, rows, columns);
      float[] first = new float[rows];
      float[] hot = new float[rows];
      float[] batchInput = new float[2 * columns + 2];
      float[] batchOutput = new float[2 * rows + 2];
      System.arraycopy(input, 0, batchInput, 1, columns);
      System.arraycopy(input, 0, batchInput, 1 + columns, columns);
      matrix.multiply(input, first);
      for (int iteration = 0; iteration < 256; iteration++) {
        matrix.multiply(input, hot);
        assertBits(first, hot, 0, "matvec iteration " + iteration);
        matrix.multiplyBatch(batchInput, 1, 2, batchOutput, 1);
        assertBits(first, batchOutput, 1, "batch first token iteration " + iteration);
        assertBits(first, batchOutput, 1 + rows, "batch second token iteration " + iteration);
      }
      for (int batchSize : new int[] {3, 4, 5, 8, 13}) {
        float[] tiledInput = new float[batchSize * columns + 2];
        float[] tiledOutput = new float[batchSize * rows + 2];
        float[][] expected = new float[batchSize][rows];
        float[] one = new float[columns];
        for (int token = 0; token < batchSize; token++) {
          for (int column = 0; column < columns; column++) {
            one[column] = input[column] + token * 0.001f;
            tiledInput[1 + token * columns + column] = one[column];
          }
          matrix.multiply(one, expected[token]);
        }
        for (int iteration = 0; iteration < 32; iteration++) {
          matrix.multiplyBatch(tiledInput, 1, batchSize, tiledOutput, 1);
          for (int token = 0; token < batchSize; token++) {
            assertBits(
                expected[token],
                tiledOutput,
                1 + token * rows,
                "batch " + batchSize + " token " + token + " iteration " + iteration);
          }
        }
      }
    }
  }

  private static void assertBits(float[] expected, float[] actual, int offset, String label) {
    for (int i = 0; i < expected.length; i++) {
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[offset + i]))
        throw new AssertionError(
            label + " row " + i + ": cold=" + expected[i] + ", actual=" + actual[offset + i]);
    }
  }
}
