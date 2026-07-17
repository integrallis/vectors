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

/** Fresh-JVM probe used by {@link Q8ColdHotDeterminismTest}. */
final class Q8ColdHotDeterminismProbe {

  private static final int ROWS = 4_096;
  private static final int COLS = 3_584;
  private static final int BATCH_SIZE = 4;
  private static final int Q8_0_BLOCK_BYTES = 34;

  private Q8ColdHotDeterminismProbe() {}

  public static void main(String[] args) {
    Random random = new Random(0xD37E_8B48L);
    float[] query = new float[COLS];
    for (int index = 0; index < query.length; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[ROWS * (COLS / 32) * Q8_0_BLOCK_BYTES];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += Q8_0_BLOCK_BYTES) {
      buffer.putShort(offset, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
    }

    PanamaVectorUtilSupport provider = new PanamaVectorUtilSupport();
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    byte[] quants = new byte[COLS];
    float[] scales = new float[COLS / 32];
    float[] first = new float[ROWS];
    float[] current = new float[ROWS];

    provider.ggufQ8_0Q8_0MatVecDot(query, weightSegment, ROWS, COLS, first, quants, scales);
    for (int iteration = 0; iteration < 8; iteration++) {
      provider.ggufQ8_0Q8_0MatVecDot(query, weightSegment, ROWS, COLS, current, quants, scales);
    }
    assertBitIdentical("Q8_0 GEMV", first, current);

    float[] queries = new float[BATCH_SIZE * COLS];
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int col = 0; col < COLS; col++) {
        queries[batch * COLS + col] = random.nextFloat() * (batch + 1.0f) - batch * 0.5f;
      }
    }
    byte[] batchQuants = new byte[BATCH_SIZE * COLS];
    float[] batchScales = new float[BATCH_SIZE * (COLS / 32)];
    float[] firstBatch = new float[BATCH_SIZE * ROWS];
    float[] currentBatch = new float[BATCH_SIZE * ROWS];

    provider.ggufQ8_0Q8_0BatchedMatmul(
        queries, weightSegment, BATCH_SIZE, ROWS, COLS, firstBatch, batchQuants, batchScales);
    for (int iteration = 0; iteration < 8; iteration++) {
      provider.ggufQ8_0Q8_0BatchedMatmul(
          queries, weightSegment, BATCH_SIZE, ROWS, COLS, currentBatch, batchQuants, batchScales);
    }
    assertBitIdentical("Q8_0 batched matmul", firstBatch, currentBatch);
  }

  private static void assertBitIdentical(String operation, float[] first, float[] current) {
    if (Arrays.equals(first, current)) {
      return;
    }
    for (int index = 0; index < first.length; index++) {
      if (Float.floatToRawIntBits(first[index]) != Float.floatToRawIntBits(current[index])) {
        throw new AssertionError(
            operation
                + " cold/hot mismatch at index "
                + index
                + ": first="
                + first[index]
                + ", hot="
                + current[index]);
      }
    }
  }
}
