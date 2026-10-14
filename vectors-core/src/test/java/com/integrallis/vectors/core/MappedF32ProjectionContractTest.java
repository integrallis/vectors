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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Mapped GGUF weights retain byte order, four-row grouping, tails and arena lifetime. */
class MappedF32ProjectionContractTest {
  @TempDir Path temporary;
  private static final float CANARY = -8765.25f;

  static Stream<Arguments> shapes() {
    return Stream.of(
            new int[] {1, 1},
            new int[] {4, 3},
            new int[] {5, 7},
            new int[] {8, 8},
            new int[] {9, 17},
            new int[] {5, 37},
            new int[] {4, 0},
            new int[] {0, 7})
        .map(s -> Arguments.of(s[0], s[1]));
  }

  @ParameterizedTest(name = "rows={0}, columns={1}")
  @MethodSource("shapes")
  void readOnlyUnalignedMappedRowsMatchIndependentDenseProducts(int rows, int columns)
      throws Exception {
    int padding = 9;
    byte[] bytes = new byte[padding + rows * columns * Float.BYTES + 13];
    Arrays.fill(bytes, (byte) 0x5a);
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    float[] query = new float[columns], expected = new float[rows];
    for (int c = 0; c < columns; c++) query[c] = (c % 7 - 3) * .25f;
    for (int r = 0; r < rows; r++) {
      double exact = 0;
      for (int c = 0; c < columns; c++) {
        float weight = ((r * 11 + c * 3) % 19 - 9) * .5f;
        buffer.putFloat(padding + (r * columns + c) * Float.BYTES, weight);
        exact += (double) query[c] * weight;
      }
      expected[r] = (float) exact;
    }
    Path path = temporary.resolve("weights.gguf");
    Files.write(path, bytes);
    MemorySegment expired;
    try (Arena arena = Arena.ofConfined();
        FileChannel file = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment mapped = file.map(FileChannel.MapMode.READ_ONLY, 0, bytes.length, arena);
      MemorySegment weights = mapped.asSlice(padding, (long) rows * columns * Float.BYTES);
      expired = weights;
      assertThat(weights.isMapped()).isTrue();
      assertThat(weights.isReadOnly()).isTrue();
      float[] out = new float[rows + 3];
      Arrays.fill(out, CANARY);
      VectorUtil.ggufF32BatchDotProduct(query, weights, rows, columns, out);
      assertThat(Arrays.copyOf(out, rows)).containsExactly(expected);
      assertThat(Arrays.copyOfRange(out, rows, out.length)).containsOnly(CANARY);
      assertThat(Files.readAllBytes(path)).containsExactly(bytes);
      float[] scalar = new float[rows];
      VectorizationProvider.newScalarProvider()
          .ggufF32MatVecDot(query, weights, rows, columns, scalar);
      assertThat(scalar).containsExactly(expected);
    }
    if (rows * columns > 0)
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufF32BatchDotProduct(query, expired, rows, columns, new float[rows]))
          .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void invalidGeometryCannotOverwriteDestination() {
    float[] query = {1, 2, 3};
    MemorySegment weights = MemorySegment.ofArray(new byte[3 * 4 * 4]);
    for (int[] shape : new int[][] {{-1, 3}, {3, -1}, {4, 4}, {4, 3}, {Integer.MAX_VALUE, 3}}) {
      float[] out = {CANARY, CANARY, CANARY};
      assertThatThrownBy(
              () -> VectorUtil.ggufF32BatchDotProduct(query, weights, shape[0], shape[1], out))
          .isInstanceOf(IllegalArgumentException.class);
      assertThat(out).containsOnly(CANARY);
    }
    float[] out = {CANARY, CANARY, CANARY};
    assertThatThrownBy(
            () -> VectorUtil.ggufF32BatchDotProduct(query, weights.asSlice(0, 35), 3, 3, out))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(out).containsOnly(CANARY);
  }
}
