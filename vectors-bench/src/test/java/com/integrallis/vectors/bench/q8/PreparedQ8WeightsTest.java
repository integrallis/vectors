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
package com.integrallis.vectors.bench.q8;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Test;

class PreparedQ8WeightsTest {

  private static final int BATCH_SIZE = 3;
  private static final int ROWS = 7;
  private static final int COLS = 96;

  @Test
  void preparedWeightsMatchTheInterleavedVectorUtilKernel() {
    Random random = new Random(0x510A7L);
    byte[] interleavedWeights = randomQ8Blocks(random, ROWS * (COLS / 32) * 34);
    float[] queries = randomQueries(random, BATCH_SIZE * COLS);
    MemorySegment weightSegment = MemorySegment.ofArray(interleavedWeights);

    float[] expected = new float[BATCH_SIZE * ROWS];
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries,
        weightSegment,
        BATCH_SIZE,
        ROWS,
        COLS,
        expected,
        new byte[BATCH_SIZE * COLS],
        new float[BATCH_SIZE * (COLS / 32)]);

    PreparedQ8Weights prepared = PreparedQ8Weights.from(weightSegment, ROWS, COLS);
    float[] actual = new float[BATCH_SIZE * ROWS];
    prepared.multiply(
        queries,
        BATCH_SIZE,
        actual,
        new byte[BATCH_SIZE * COLS],
        new float[BATCH_SIZE * (COLS / 32)]);

    assertThat(actual).containsExactly(expected);
    assertThat(prepared.quantByteCount()).isEqualTo(ROWS * COLS);
    assertThat(prepared.scaleCount()).isEqualTo(ROWS * (COLS / 32));
  }

  private static byte[] randomQ8Blocks(Random random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < blocks.length; offset += 34) {
      float scale = 0.001f + random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }
    return blocks;
  }

  private static float[] randomQueries(Random random, int length) {
    float[] queries = new float[length];
    for (int index = 0; index < length; index++) {
      queries[index] = random.nextFloat() * 4.0f - 2.0f;
    }
    return queries;
  }
}
