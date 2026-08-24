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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BFloat16MatrixTest {

  @Test
  void multipliesKnownBfloat16BitPatternsWithFloatAccumulation() {
    BFloat16Matrix matrix =
        BFloat16Matrix.of(
            MemorySegment.ofArray(bf16Bits(0x3f80, 0xc000, 0x3f00, 0x4040, 0xbf80, 0x4000)), 2, 3);
    float[] output = new float[2];

    matrix.multiply(new float[] {0.25f, -0.5f, 4.0f}, output);

    assertThat(output).containsExactly(3.25f, 9.25f);
  }

  @Test
  void preservesSubnormalAndRoundedReferenceValues() {
    BFloat16Matrix matrix =
        BFloat16Matrix.of(
            MemorySegment.ofArray(bf16Bits(0x3f81, 0xbf7f, 0x3eab, 0x0001, 0x8001, 0x4120)), 2, 3);
    float[] input = {1.25f, -2.5f, 0.75f};
    float[] output = new float[2];

    matrix.multiply(input, output);

    assertThat(output[0]).isCloseTo(4.0004883f, within(1.0e-6f));
    assertThat(output[1]).isCloseTo(7.5f, within(1.0e-6f));
  }

  @Test
  void slicesRowsWithoutCopyingAndReadsIndividualValues() {
    BFloat16Matrix matrix =
        BFloat16Matrix.of(
            MemorySegment.ofArray(bf16Bits(0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0)), 3, 2);
    BFloat16Matrix slice = matrix.rowSlice(1, 2);
    float[] output = new float[2];

    slice.multiply(new float[] {1.0f, 0.5f}, output);

    assertThat(slice.rows()).isEqualTo(2);
    assertThat(slice.columns()).isEqualTo(2);
    assertThat(slice.value(0, 0)).isEqualTo(3.0f);
    assertThat(output).containsExactly(5.0f, 8.0f);
  }

  @Test
  void readsFromMappedStorage(@TempDir Path directory) throws IOException {
    byte[] data = bf16Bits(0x3f80, 0x4000, 0x4040, 0x4080);
    Path path = directory.resolve("weights.bf16");
    Files.write(path, data);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, data.length, arena);
      BFloat16Matrix matrix = BFloat16Matrix.of(mapped, 2, 2);
      float[] output = new float[2];

      matrix.multiply(new float[] {2.0f, -1.0f}, output);

      assertThat(output).containsExactly(0.0f, 2.0f);
    }
  }

  @Test
  void multipliesCompleteVectorBlocksAndScalarTail() {
    int columns = 37;
    int rows = 3;
    int[] bits = new int[rows * columns];
    float[] input = new float[columns];
    float[] expected = new float[rows];
    for (int column = 0; column < columns; column++) {
      input[column] = (column % 7 - 3) * 0.125f;
    }
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        float value = (row + 1) * (column % 5 - 2) * 0.25f;
        int bitPattern = Float.floatToRawIntBits(value) >>> Short.SIZE;
        bits[row * columns + column] = bitPattern;
        expected[row] += Float.intBitsToFloat(bitPattern << Short.SIZE) * input[column];
      }
    }
    BFloat16Matrix matrix = BFloat16Matrix.of(MemorySegment.ofArray(bf16Bits(bits)), rows, columns);
    float[] output = new float[rows];

    matrix.multiply(input, output);

    assertThat(output).containsExactly(expected);
  }

  @Test
  void multipliesLargeRowCountsWithSequentialMemoryAccess() {
    int rows = 8192;
    int columns = 3;
    int[] bits = new int[rows * columns];
    for (int row = 0; row < rows; row++) {
      bits[row * columns] = 0x3f80;
      bits[row * columns + 1] = row % 2 == 0 ? 0x4000 : 0xc000;
      bits[row * columns + 2] = 0x3f00;
    }
    BFloat16Matrix matrix = BFloat16Matrix.of(MemorySegment.ofArray(bf16Bits(bits)), rows, columns);
    float[] output = new float[rows];

    matrix.multiply(new float[] {1.0f, 0.25f, -2.0f}, output);

    assertThat(output[0]).isEqualTo(0.5f);
    assertThat(output[1]).isEqualTo(-0.5f);
    assertThat(output[rows - 1]).isEqualTo(-0.5f);
  }

  @Test
  void rejectsBadGeometryStorageAndInvocation() {
    MemorySegment valid = MemorySegment.ofArray(bf16Bits(0x3f80, 0x4000));
    BFloat16Matrix matrix = BFloat16Matrix.of(valid, 1, 2);

    assertThatThrownBy(() -> BFloat16Matrix.of(valid, 0, 2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
    assertThatThrownBy(() -> BFloat16Matrix.of(valid, 1, 3))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("bytes");
    assertThatThrownBy(() -> matrix.multiply(new float[1], new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("input");
    assertThatThrownBy(() -> matrix.multiply(new float[2], new float[0]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("output");
    assertThatThrownBy(() -> matrix.rowSlice(1, 1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> matrix.value(0, 2)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  private static byte[] bf16Bits(int... bits) {
    ByteBuffer bytes =
        ByteBuffer.allocate(bits.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int value : bits) {
      bytes.putShort((short) value);
    }
    return bytes.array();
  }
}
