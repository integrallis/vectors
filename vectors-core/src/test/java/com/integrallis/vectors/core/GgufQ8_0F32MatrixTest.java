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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufQ8_0F32MatrixTest {

  @Test
  void expandsQ8BlocksAndMultipliesBatchMajorInputsIncludingRowAndBatchTails() {
    int rows = 7;
    int columns = 96;
    int batchSize = 5;
    Random random = new Random(0xF32_080L);
    byte[] q8Weights = randomQ8Blocks(random, rows, columns);
    float[] values = randomValues(random, batchSize * columns);
    float[] input = withSentinels(values, 2, 3, -91.0f);
    float[] expected = scalarDecodedMatmul(q8Weights, values, batchSize, rows, columns);
    float[] output = new float[1 + expected.length + 2];
    java.util.Arrays.fill(output, 73.0f);

    GgufQ8_0F32Matrix matrix =
        GgufQ8_0F32Matrix.from(MemorySegment.ofArray(q8Weights), rows, columns);
    matrix.multiplyBatch(input, 2, batchSize, output, 1);

    assertThat(matrix.rows()).isEqualTo(rows);
    assertThat(matrix.columns()).isEqualTo(columns);
    assertThat(output[0]).isEqualTo(73.0f);
    assertThat(output[output.length - 1]).isEqualTo(73.0f);
    assertThat(output[output.length - 2]).isEqualTo(73.0f);
    for (int index = 0; index < expected.length; index++) {
      float tolerance = Math.max(1.0e-4f, Math.abs(expected[index]) * 2.0e-5f);
      assertThat(output[index + 1])
          .as("batch-major output %s", index)
          .isCloseTo(expected[index], within(tolerance));
    }
  }

  @Test
  void reportsSerializedAndExpandedStorageCosts() {
    int rows = 7;
    int columns = 96;
    byte[] q8Weights = randomQ8Blocks(new Random(1L), rows, columns);

    GgufQ8_0F32Matrix matrix =
        GgufQ8_0F32Matrix.from(MemorySegment.ofArray(q8Weights), rows, columns);

    assertThat(matrix.serializedByteCount()).isEqualTo(rows * (columns / 32) * 34L);
    assertThat(matrix.expandedByteCount()).isEqualTo(rows * columns * Float.BYTES);
    assertThat(matrix.expansionRatio()).isCloseTo(64.0 / 17.0, within(1.0e-12));
  }

  @Test
  void ownsTheExpandedExecutionLayoutInsteadOfRetainingTheSourceSegment() {
    byte[] q8Weights = q8Block(0.5f, (byte) 2);
    GgufQ8_0F32Matrix matrix = GgufQ8_0F32Matrix.from(MemorySegment.ofArray(q8Weights), 1, 32);
    java.util.Arrays.fill(q8Weights, (byte) 0);
    float[] input = new float[32];
    java.util.Arrays.fill(input, 1.0f);
    float[] output = new float[1];

    matrix.multiplyBatch(input, 1, output);

    assertThat(output).containsExactly(32.0f);
  }

  @Test
  void rejectsInvalidGeometryStorageAndBatchRanges() {
    byte[] oneRow = q8Block(0.5f, (byte) 2);
    MemorySegment valid = MemorySegment.ofArray(oneRow);
    GgufQ8_0F32Matrix matrix = GgufQ8_0F32Matrix.from(valid, 1, 32);

    assertThatThrownBy(() -> GgufQ8_0F32Matrix.from(null, 1, 32))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("q8Weights");
    assertThatThrownBy(() -> GgufQ8_0F32Matrix.from(valid, 0, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
    assertThatThrownBy(() -> GgufQ8_0F32Matrix.from(valid, 1, 31))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("columns");
    assertThatThrownBy(() -> GgufQ8_0F32Matrix.from(MemorySegment.ofArray(new byte[33]), 1, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("34 bytes");
    assertThatThrownBy(() -> GgufQ8_0F32Matrix.from(MemorySegment.ofArray(new byte[35]), 1, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("34 bytes");
    assertThatThrownBy(() -> matrix.multiplyBatch(null, 1, new float[1]))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("input");
    assertThatThrownBy(() -> matrix.multiplyBatch(new float[32], 1, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("output");
    assertThatThrownBy(() -> matrix.multiplyBatch(new float[32], 0, new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchSize");
    assertThatThrownBy(() -> matrix.multiplyBatch(new float[32], 1, 1, new float[1], 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("input range");
    assertThatThrownBy(() -> matrix.multiplyBatch(new float[32], 0, 1, new float[1], 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("output range");
  }

  private static byte[] randomQ8Blocks(Random random, int rows, int columns) {
    byte[] blocks = new byte[rows * (columns / 32) * 34];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < blocks.length; offset += 34) {
      float scale = 0.001f + random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }
    return blocks;
  }

  private static byte[] q8Block(float scale, byte quant) {
    byte[] block = new byte[34];
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(Float.floatToFloat16(scale));
    java.util.Arrays.fill(block, 2, block.length, quant);
    return block;
  }

  private static float[] randomValues(Random random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = random.nextFloat() * 4.0f - 2.0f;
    }
    return values;
  }

  private static float[] withSentinels(
      float[] values, int prefixLength, int suffixLength, float sentinel) {
    float[] padded = new float[prefixLength + values.length + suffixLength];
    java.util.Arrays.fill(padded, sentinel);
    System.arraycopy(values, 0, padded, prefixLength, values.length);
    return padded;
  }

  private static float[] scalarDecodedMatmul(
      byte[] q8Weights, float[] input, int batchSize, int rows, int columns) {
    ByteBuffer buffer = ByteBuffer.wrap(q8Weights).order(ByteOrder.LITTLE_ENDIAN);
    int blocksPerRow = columns / 32;
    float[] result = new float[batchSize * rows];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
          int blockOffset = (row * blocksPerRow + block) * 34;
          float scale = Float.float16ToFloat(buffer.getShort(blockOffset));
          for (int lane = 0; lane < 32; lane++) {
            float weight = scale * buffer.get(blockOffset + Short.BYTES + lane);
            sum = Math.fma(weight, input[batch * columns + block * 32 + lane], sum);
          }
        }
        result[batch * rows + row] = sum;
      }
    }
    return result;
  }
}
