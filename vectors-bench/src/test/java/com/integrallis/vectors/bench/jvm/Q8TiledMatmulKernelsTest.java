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
package com.integrallis.vectors.bench.jvm;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Test;

class Q8TiledMatmulKernelsTest {

  @Test
  void everyTwoDimensionalTileMatchesTheProductionKernelIncludingTails() {
    int batchSize = 9;
    int rows = 7;
    int cols = 96;
    int blocks = cols / 32;
    Random random = new Random(0x28d71eL);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    MemorySegment weights = MemorySegment.ofArray(randomQ8Blocks(random, rows * blocks * 34));
    byte[] activationQuants = new byte[batchSize * cols];
    float[] activationScales = new float[batchSize * blocks];
    float[] expected = new float[batchSize * rows];
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, weights, batchSize, rows, cols, expected, activationQuants, activationScales);

    Q8TiledMatmulKernels.Workspace workspace = Q8TiledMatmulKernels.workspace(4, 8);
    for (int rowTile : new int[] {1, 2, 4}) {
      for (int batchTile : new int[] {1, 2, 4, 8}) {
        float[] actual = new float[batchSize * rows];
        Q8TiledMatmulKernels.tiled(
            weights,
            batchSize,
            rows,
            cols,
            activationQuants,
            activationScales,
            actual,
            rowTile,
            batchTile,
            workspace);

        assertThat(actual)
            .as("row tile %s, batch tile %s", rowTile, batchTile)
            .containsExactly(expected);
      }
    }
  }

  private static byte[] randomQ8Blocks(Random random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += 34) {
      buffer.putShort(offset, scale);
    }
    return blocks;
  }
}
