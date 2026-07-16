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

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

/** Fresh-JVM probe used by {@link Q4KColdHotDeterminismTest}. */
final class Q4KColdHotDeterminismProbe {

  private static final int ROWS = 4_096;
  private static final int COLS = 3_584;
  private static final int Q4_K_BLOCK_BYTES = 144;

  private Q4KColdHotDeterminismProbe() {}

  public static void main(String[] args) {
    Random random = new Random(0xD37E_4B48L);
    float[] query = new float[COLS];
    for (int index = 0; index < query.length; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[ROWS * (COLS / 256) * Q4_K_BLOCK_BYTES];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += Q4_K_BLOCK_BYTES) {
      buffer.putShort(offset, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(random.nextFloat() * 0.05f));
    }

    PanamaVectorUtilSupport provider = new PanamaVectorUtilSupport();
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    byte[] quants = new byte[COLS];
    float[] scales = new float[COLS / 256];
    short[] sums = new short[COLS / 16];
    float[] first = new float[ROWS];
    float[] current = new float[ROWS];

    provider.ggufQ4_KQ8_KMatVecDot(query, weightSegment, ROWS, COLS, first, quants, scales, sums);
    for (int iteration = 0; iteration < 8; iteration++) {
      provider.ggufQ4_KQ8_KMatVecDot(
          query, weightSegment, ROWS, COLS, current, quants, scales, sums);
    }

    if (!Arrays.equals(first, current)) {
      for (int row = 0; row < ROWS; row++) {
        if (Float.floatToRawIntBits(first[row]) != Float.floatToRawIntBits(current[row])) {
          throw new AssertionError(
              "Q4_K cold/hot mismatch at row "
                  + row
                  + ": first="
                  + first[row]
                  + ", hot="
                  + current[row]);
        }
      }
    }
  }
}
