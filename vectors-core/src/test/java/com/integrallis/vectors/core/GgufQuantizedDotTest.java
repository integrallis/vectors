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
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import java.util.function.IntUnaryOperator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufQuantizedDotTest {

  @Test
  void f32BatchDotProduct_readsLittleEndianMappedRows() {
    float[] query = {1.5f, -2.0f, 0.25f, 4.0f, -0.5f};
    float[] matrix = {
      0.5f, 1.0f, -3.0f, 0.25f, 2.0f,
      -1.0f, 0.5f, 2.0f, -0.75f, 4.0f
    };
    float[] out = new float[2];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, f32Bytes(matrix));

      VectorUtil.ggufF32BatchDotProduct(query, segment, 2, 5, out);

      assertThat(out[0]).isCloseTo(-2.0f, within(1e-5f));
      assertThat(out[1]).isCloseTo(-7.0f, within(1e-5f));
    }
  }

  @Test
  void f32BatchDotProduct_rejectsTruncatedMatrix() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocate(7 * Float.BYTES);

      assertThatThrownBy(
              () ->
                  VectorUtil.ggufF32BatchDotProduct(
                      new float[] {1, 2, 3, 4}, segment, 2, 4, new float[2]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("byteSize");
    }
  }

  @Test
  void q4_0DotProduct_matchesDecodedReference() {
    float[] query = new float[32];
    byte[] block = q4Block(0.5f, query, null);

    float expected = 0f;
    for (int i = 0; i < query.length; i++) {
      query[i] = (i % 7) - 3.0f;
      int packed = block[2 + (i & 0x0F)] & 0xFF;
      int nibble = i < 16 ? packed & 0x0F : (packed >>> 4) & 0x0F;
      expected = Math.fma(query[i], (nibble - 8) * 0.5f, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ4_0DotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-5f));
    }
  }

  @Test
  void q4_0DotProduct_usesGgmlSplitNibbleLayout() {
    float[] query = new float[32];
    query[16] = 1.0f;
    byte[] block = q4Block(0.5f, query, (lo, hi) -> lo == 0 ? 0x98 : 0x88);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      assertThat(VectorUtil.ggufQ4_0DotProduct(query, segment, 0, query.length))
          .isCloseTo(0.5f, within(1e-5f));
    }
  }

  @Test
  void q8_0DotProduct_matchesDecodedReference() {
    float[] query = new float[32];
    byte[] block = q8Block(0.25f);

    float expected = 0f;
    for (int i = 0; i < query.length; i++) {
      query[i] = (i % 5) * 0.75f - 1.5f;
      expected = Math.fma(query[i], block[2 + i] * 0.25f, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ8_0DotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-5f));
    }
  }

  @Test
  void q6_KDotProduct_matchesDecodedReference() {
    float[] query = patternedQuery(256);
    byte[] block = q6KBlock(0.125f, i -> (i % 64) - 32, i -> (i % 7) - 3);

    float expected = 0f;
    for (int i = 0; i < query.length; i++) {
      int quant = (i % 64) - 32;
      int subScale = (i / 16) % 7 - 3;
      expected = Math.fma(query[i], 0.125f * subScale * quant, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ6_KDotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-5f));
    }
  }

  @Test
  void q6_KLongOffsetRowsMatchTheEstablishedMappedKernelExactly() throws IOException {
    int rows = 2;
    int cols = 512;
    byte[] firstRow = repeat(q6KBlock(0.125f, i -> (i * 5 + 3) % 64 - 32, i -> i - 8), 2);
    byte[] secondRow = repeat(q6KBlock(-0.25f, i -> 31 - (i % 64), i -> 7 - i), 2);
    byte[] matrix = concat(firstRow, secondRow);
    float[] query = patternedQuery(cols);
    float[] expected = new float[rows];
    byte[] q8Quants = new byte[cols];
    float[] q8Scales = new float[cols / 256];
    Path path = Files.createTempFile("vectors-q6-k-", ".bin");
    Files.write(path, matrix);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, matrix.length, arena);
      assertThat(mapped.isMapped()).isTrue();

      VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
          query, mapped, rows, cols, expected, q8Quants, q8Scales);
      GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, null);

      long rowBytes = firstRow.length;
      float[] actual = {
        PanamaVectorUtilSupport.ggufQ6_KQ8_KLongOffsetRowDot(
            mapped, 0, rowBytes, q8Quants, q8Scales),
        PanamaVectorUtilSupport.ggufQ6_KQ8_KLongOffsetRowDot(
            mapped, rowBytes, rowBytes, q8Quants, q8Scales)
      };

      assertThat(actual).containsExactly(expected);
    } finally {
      Files.deleteIfExists(path);
    }
  }

  @Test
  void q4_KLongOffsetRowsMatchTheEstablishedMappedKernelExactly() throws IOException {
    int rows = 2;
    int cols = 512;
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstRow = repeat(q4KBlock(0.125f, 0.0625f, i -> (i * 5 + 3) & 0x0F, scales, mins), 2);
    byte[] secondRow = repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - (i & 0x0F), scales, mins), 2);
    byte[] matrix = concat(firstRow, secondRow);
    float[] query = patternedQuery(cols);
    float[] expected = new float[rows];
    byte[] q8Quants = new byte[cols];
    float[] q8Scales = new float[cols / 256];
    short[] q8Sums = new short[cols / 16];
    Path path = Files.createTempFile("vectors-q4-k-", ".bin");
    Files.write(path, matrix);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, matrix.length, arena);
      assertThat(mapped.isMapped()).isTrue();

      VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
          query, MemorySegment.ofArray(matrix), rows, cols, expected, q8Quants, q8Scales, q8Sums);
      GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

      long rowBytes = firstRow.length;
      float[] actual = {
        PanamaVectorUtilSupport.ggufQ4_KQ8_KLongOffsetRowDot(
            mapped, 0, rowBytes, q8Quants, q8Scales, q8Sums),
        PanamaVectorUtilSupport.ggufQ4_KQ8_KLongOffsetRowDot(
            mapped, rowBytes, rowBytes, q8Quants, q8Scales, q8Sums)
      };

      assertThat(actual).containsExactly(expected);
    } finally {
      Files.deleteIfExists(path);
    }
  }

  @Test
  void q4_KLongOffsetBatchedMatmulMatchesTheEstablishedKernelExactly() throws IOException {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstRow = repeat(q4KBlock(0.125f, 0.0625f, i -> (i * 5 + 3) & 0x0F, scales, mins), 2);
    byte[] secondRow = repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - (i & 0x0F), scales, mins), 2);
    byte[] matrix = concat(firstRow, secondRow);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = (float) Math.sin((index + 0.25) * 0.03125);
    }
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    byte[] q8Quants = new byte[batchSize * cols];
    float[] q8Scales = new float[batchSize * (cols / 256)];
    short[] q8Sums = new short[batchSize * (cols / 16)];
    Path path = Files.createTempFile("vectors-q4-k-batch-", ".bin");
    Files.write(path, matrix);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, matrix.length, arena);
      assertThat(mapped.isMapped()).isTrue();

      VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
          queries,
          MemorySegment.ofArray(matrix),
          batchSize,
          rows,
          cols,
          expected,
          q8Quants,
          q8Scales,
          q8Sums);
      PanamaVectorUtilSupport.ggufQ4_KQ8_KLongOffsetBatchedMatmul(
          mapped, batchSize, rows, cols, actual, q8Quants, q8Scales, q8Sums);

      assertThat(actual).containsExactly(expected);
    } finally {
      Files.deleteIfExists(path);
    }
  }

  @Test
  void q5_KLongOffsetRowsMatchTheEstablishedMappedKernelExactly() throws IOException {
    int rows = 2;
    int cols = 512;
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstRow = repeat(q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) & 0x1F, scales, mins), 2);
    byte[] secondRow = repeat(q5KBlock(-0.25f, 0.03125f, i -> 31 - (i & 0x1F), scales, mins), 2);
    byte[] matrix = concat(firstRow, secondRow);
    float[] query = patternedQuery(cols);
    float[] expected = new float[rows];
    byte[] q8Quants = new byte[cols];
    float[] q8Scales = new float[cols / 256];
    short[] q8Sums = new short[cols / 16];
    Path path = Files.createTempFile("vectors-q5-k-", ".bin");
    Files.write(path, matrix);

    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, matrix.length, arena);
      assertThat(mapped.isMapped()).isTrue();

      VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
          query, mapped, rows, cols, expected, q8Quants, q8Scales, q8Sums);
      GgufQuantizationSupport.quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

      long rowBytes = firstRow.length;
      float[] actual = {
        PanamaVectorUtilSupport.ggufQ5_KQ8_KLongOffsetRowDot(
            mapped, 0, rowBytes, q8Quants, q8Scales, q8Sums),
        PanamaVectorUtilSupport.ggufQ5_KQ8_KLongOffsetRowDot(
            mapped, rowBytes, rowBytes, q8Quants, q8Scales, q8Sums)
      };

      assertThat(actual).containsExactly(expected);
    } finally {
      Files.deleteIfExists(path);
    }
  }

  @Test
  void q6_KDequantize_matchesDecodedReferenceAtOutputOffset() {
    byte[] block = q6KBlock(0.125f, i -> (i % 64) - 32, i -> (i % 7) - 3);
    float[] decoded = new float[258];
    decoded[0] = 99.0f;
    decoded[257] = 98.0f;

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      VectorUtil.ggufQ6_KDequantize(segment, 0, decoded, 1, 256);
    }

    assertThat(decoded[0]).isEqualTo(99.0f);
    assertThat(decoded[257]).isEqualTo(98.0f);
    for (int index = 0; index < 256; index++) {
      int quant = (index % 64) - 32;
      int subScale = (index / 16) % 7 - 3;
      assertThat(decoded[index + 1]).isCloseTo(0.125f * subScale * quant, within(1e-6f));
    }
  }

  @Test
  void q4_KDotProduct_matchesDecodedReferenceWithPackedScalesAndMins() {
    float[] query = patternedQuery(256);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] block = q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins);

    float expected = 0f;
    for (int i = 0; i < query.length; i++) {
      int group = i / 32;
      float weight = 0.125f * scales[group] * (i % 16) - 0.0625f * mins[group];
      expected = Math.fma(query[i], weight, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ4_KDotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-4f));
    }
  }

  @Test
  void q4_KDequantize_matchesDecodedReferenceAtOutputOffset() {
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] block = q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins);
    float[] decoded = new float[258];
    decoded[0] = 99.0f;
    decoded[257] = 98.0f;

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      VectorUtil.ggufQ4_KDequantize(segment, 0, decoded, 1, 256);
    }

    assertThat(decoded[0]).isEqualTo(99.0f);
    assertThat(decoded[257]).isEqualTo(98.0f);
    for (int index = 0; index < 256; index++) {
      int group = index / 32;
      float expected = 0.125f * scales[group] * (index % 16) - 0.0625f * mins[group];
      assertThat(decoded[index + 1]).isCloseTo(expected, within(1e-6f));
    }
  }

  @Test
  void q5_KDotProduct_matchesDecodedReferenceWithHighBitsScalesAndMins() {
    float[] query = patternedQuery(256);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] block = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);

    float expected = 0.0f;
    for (int index = 0; index < query.length; index++) {
      int group = index / 32;
      int quant = (index * 7 + 3) % 32;
      float weight = 0.125f * scales[group] * quant - 0.0625f * mins[group];
      expected = Math.fma(query[index], weight, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ5_KDotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-3f));
    }
  }

  @Test
  void q5_KDequantize_matchesDecodedReferenceAtOutputOffset() {
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] block = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);
    float[] decoded = new float[258];
    decoded[0] = 99.0f;
    decoded[257] = 98.0f;

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      VectorUtil.ggufQ5_KDequantize(segment, 0, decoded, 1, 256);
    }

    assertThat(decoded[0]).isEqualTo(99.0f);
    assertThat(decoded[257]).isEqualTo(98.0f);
    for (int index = 0; index < 256; index++) {
      int group = index / 32;
      int quant = (index * 7 + 3) % 32;
      float expected = 0.125f * scales[group] * quant - 0.0625f * mins[group];
      assertThat(decoded[index + 1]).isCloseTo(expected, within(1e-6f));
    }
  }

  @Test
  void q5_0DotProduct_matchesDecodedReferenceWithHighBitMask() {
    float[] query = patternedQuery(32);
    byte[] block = q5Block(0.25f, i -> (i * 7) % 32 - 16);

    float expected = 0.0f;
    for (int i = 0; i < query.length; i++) {
      expected = Math.fma(query[i], 0.25f * ((i * 7) % 32 - 16), expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ5_0DotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-5f));
    }
  }

  @Test
  void q4_0BatchDotProduct_respectsRowOffsets() {
    float[] query = ones(32);
    byte[] row0 = q4Block(1.0f, query, (lo, hi) -> 0x98);
    byte[] row1 = q4Block(2.0f, query, (lo, hi) -> 0x87);
    byte[] matrix = concat(row0, row1);
    float[] out = new float[2];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, matrix);

      VectorUtil.ggufQ4_0BatchDotProduct(query, segment, 2, 32, out);

      assertThat(out[0])
          .isCloseTo(VectorUtil.ggufQ4_0DotProduct(query, segment, 0, 32), within(1e-5f));
      assertThat(out[1])
          .isCloseTo(VectorUtil.ggufQ4_0DotProduct(query, segment, 18, 32), within(1e-5f));
    }
  }

  @Test
  void q4_0Q8_0BatchDotProduct_quantizesEachQueryBlockUsingGgmlSemantics() {
    float[] query = new float[32];
    query[0] = 1.0f;
    query[1] = 0.49f;
    byte[] row0 = q4Block(1.0f, query, (lo, hi) -> 0x99);
    byte[] row1 = q4Block(1.0f, query, (lo, hi) -> 0xAA);
    float[] out = new float[2];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[query.length / 32];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
          query, segment, 2, query.length, out, q8Quants, q8Scales, q4Corrections(query.length));

      float q8Scale = Float.float16ToFloat(Float.floatToFloat16(1.0f / 127.0f));
      assertThat(out[0]).isCloseTo(189.0f * q8Scale, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f * q8Scale, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
    }
  }

  @Test
  void q4_0Q8_0ExplicitKernelModesMatchExactlyForModelSizedRows() {
    int cols = 1024;
    int rows = 3;
    float[] query = patternedQuery(cols);
    byte[] block = q4Block(0.125f, ones(cols), (low, high) -> (low * 11 + high * 7) & 0xFF);
    byte[] weights = repeat(block, rows * (cols / 32));
    float[] widened = new float[rows];
    float[] pairwise = new float[rows];
    float[] unsignedPairwise = new float[rows];

    MemorySegment segment = MemorySegment.ofArray(weights);
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        segment,
        rows,
        cols,
        widened,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols),
        GgufQ4Kernel.WIDENED);
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        segment,
        rows,
        cols,
        pairwise,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols),
        GgufQ4Kernel.SHORT_PAIRWISE);
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        segment,
        rows,
        cols,
        unsignedPairwise,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols),
        GgufQ4Kernel.UNSIGNED_PAIRWISE);

    assertThat(pairwise).containsExactly(widened);
    assertThat(unsignedPairwise).containsExactly(widened);
  }

  @Test
  void q4_0Q8_0DualBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 1024;
    int firstRows = 2;
    int secondRows = 3;
    float[] query = patternedQuery(cols);
    byte[] firstWeights =
        concat(
            repeat(q4Block(0.125f, ones(cols), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF), cols / 32),
            repeat(q4Block(-0.25f, ones(cols), (lo, hi) -> (lo * 7 + hi) & 0xFF), cols / 32));
    byte[] secondWeights =
        concat(
            concat(
                repeat(q4Block(0.5f, ones(cols), (lo, hi) -> (lo + hi * 11) & 0xFF), cols / 32),
                repeat(
                    q4Block(-0.0625f, ones(cols), (lo, hi) -> (lo * 13 + hi) & 0xFF), cols / 32)),
            repeat(q4Block(0.03125f, ones(cols), (lo, hi) -> (lo * 3 + hi * 7) & 0xFF), cols / 32));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];

    MemorySegment firstSegment = MemorySegment.ofArray(firstWeights);
    MemorySegment secondSegment = MemorySegment.ofArray(secondWeights);
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        firstSegment,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols));
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        secondSegment,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols));

    VectorUtil.ggufQ4_0Q8_0DualBatchDotProduct(
        query,
        firstSegment,
        firstRows,
        actualFirst,
        secondSegment,
        secondRows,
        actualSecond,
        cols,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols),
        GgufQ4Kernel.SHORT_PAIRWISE);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q4_0Q8_0TripleBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 1024;
    int firstRows = 4;
    int secondRows = 2;
    int thirdRows = 3;
    float[] query = patternedQuery(cols);
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(
                q4Block(0.125f, ones(cols), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF),
                firstRows * (cols / 32)));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(
                q4Block(-0.25f, ones(cols), (lo, hi) -> (lo * 7 + hi) & 0xFF),
                secondRows * (cols / 32)));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(
            repeat(
                q4Block(0.03125f, ones(cols), (lo, hi) -> (lo * 3 + hi * 7) & 0xFF),
                thirdRows * (cols / 32)));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] expectedThird = new float[thirdRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];
    float[] actualThird = new float[thirdRows];

    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols));
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols));
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        thirdWeight,
        thirdRows,
        cols,
        expectedThird,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols));

    VectorUtil.ggufQ4_0Q8_0TripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        cols,
        new byte[cols],
        new float[cols / 32],
        q4Corrections(cols),
        GgufQ4Kernel.SHORT_PAIRWISE);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q4_0Q8_0BatchedMatmulMatchesIndependentQueriesNumerically() {
    int batchSize = 3;
    int rows = 2;
    int cols = 1024;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] = ((batch + 1) * (col - 13)) / 17.0f;
      }
    }
    byte[] row0 =
        repeat(q4Block(0.125f, ones(cols), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF), cols / 32);
    byte[] row1 = repeat(q4Block(-0.25f, ones(cols), (lo, hi) -> (lo * 7 + hi) & 0xFF), cols / 32);
    float[] expected = new float[batchSize * rows];
    float[] unsignedExpected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] unsignedActual = new float[batchSize * rows];
    byte[] q8Quants = new byte[batchSize * cols];
    float[] q8Scales = new float[batchSize * (cols / 32)];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));
      for (int batch = 0; batch < batchSize; batch++) {
        float[] query = new float[cols];
        System.arraycopy(queries, batch * cols, query, 0, cols);
        float[] result = new float[rows];
        VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
            query,
            segment,
            rows,
            cols,
            result,
            new byte[cols],
            new float[cols / 32],
            q4Corrections(cols));
        System.arraycopy(result, 0, expected, batch * rows, rows);
        VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
            query,
            segment,
            rows,
            cols,
            result,
            new byte[cols],
            new float[cols / 32],
            q4Corrections(cols),
            GgufQ4Kernel.UNSIGNED_PAIRWISE);
        System.arraycopy(result, 0, unsignedExpected, batch * rows, rows);
      }

      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          queries,
          segment,
          batchSize,
          rows,
          cols,
          actual,
          q8Quants,
          q8Scales,
          q4Corrections(batchSize * cols),
          GgufQ4Kernel.SHORT_PAIRWISE);
      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          queries,
          segment,
          batchSize,
          rows,
          cols,
          unsignedActual,
          new byte[batchSize * cols],
          new float[batchSize * (cols / 32)],
          q4Corrections(batchSize * cols),
          GgufQ4Kernel.UNSIGNED_PAIRWISE);

      assertThat(actual).containsExactly(expected);
      for (int index = 0; index < unsignedActual.length; index++) {
        assertThat(unsignedActual[index]).isCloseTo(unsignedExpected[index], within(0.01f));
      }
      assertThat(q8Quants[0]).isNotZero();
    }
  }

  @Test
  void q4_0PrequantizedRowRangesComposeToTheExactBatchedResult() {
    int batchSize = 3;
    int rows = 5;
    int cols = 1024;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow =
        repeat(q4Block(0.125f, ones(32), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF), cols / 32);
    byte[] secondRow =
        repeat(q4Block(-0.25f, ones(32), (lo, hi) -> (lo * 7 + hi) & 0xFF), cols / 32);
    MemorySegment weight =
        MemorySegment.ofArray(
            concat(concat(firstRow, secondRow), concat(concat(firstRow, secondRow), firstRow)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(batchSize, cols);

    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        weight,
        batchSize,
        rows,
        cols,
        expected,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 32)],
        q4Corrections(batchSize * cols),
        new float[batchSize * rows * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);
    activation.quantizeForQ4(queries, batchSize, GgufQ4Kernel.UNSIGNED_PAIRWISE);

    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        weight,
        batchSize,
        rows,
        cols,
        0,
        2,
        actual,
        activation,
        new float[batchSize * rows * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        weight,
        batchSize,
        rows,
        cols,
        2,
        rows,
        actual,
        activation,
        new float[batchSize * rows * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q4_0BlockRangeActivationQuantizationMatchesWholeBatchExactly() {
    int batchSize = 3;
    int rows = 5;
    int cols = 1024;
    int blocks = cols / 32;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] row = repeat(q4Block(0.125f, ones(32), (lo, hi) -> (lo * 11 + hi * 3) & 0xFF), blocks);
    MemorySegment weight = MemorySegment.ofArray(repeat(row, rows));
    GgufQ8_0Batch whole = GgufQ8_0Batch.allocate(batchSize, cols);
    GgufQ8_0Batch rangeWise = GgufQ8_0Batch.allocate(batchSize, cols);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    whole.quantizeForQ4(queries, batchSize, GgufQ4Kernel.UNSIGNED_PAIRWISE);
    int split = blocks / 3;
    rangeWise.quantizeBlockRangeForQ4(queries, batchSize, 0, split, GgufQ4Kernel.UNSIGNED_PAIRWISE);
    rangeWise.quantizeBlockRangeForQ4(
        queries, batchSize, split, blocks, GgufQ4Kernel.UNSIGNED_PAIRWISE);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        weight,
        batchSize,
        rows,
        cols,
        0,
        rows,
        expected,
        whole,
        new float[batchSize * rows * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
        weight,
        batchSize,
        rows,
        cols,
        0,
        rows,
        actual,
        rangeWise,
        new float[batchSize * rows * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0BatchRejectsInvalidShapeAndBlockRanges() {
    assertThatThrownBy(() -> GgufQ8_0Batch.allocate(0, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchCapacity");
    assertThatThrownBy(() -> GgufQ8_0Batch.allocate(1, 33))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimensions");

    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(1, 32);
    assertThatThrownBy(
            () ->
                activation.quantizeBlockRangeForQ4(
                    new float[32], 1, 1, 2, GgufQ4Kernel.UNSIGNED_PAIRWISE))
        .isInstanceOf(IndexOutOfBoundsException.class)
        .hasMessageContaining("block range");
  }

  @Test
  void q4_0Q8_0DualBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 1024;
    int firstRows = 2;
    int secondRows = 3;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow =
        repeat(q4Block(0.125f, ones(32), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF), cols / 32);
    byte[] secondRow =
        repeat(q4Block(-0.25f, ones(32), (lo, hi) -> (lo * 7 + hi) & 0xFF), cols / 32);
    MemorySegment firstWeight = MemorySegment.ofArray(repeat(firstRow, firstRows));
    MemorySegment secondWeight = MemorySegment.ofArray(repeat(secondRow, secondRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];
    byte[] quants = new byte[batchSize * cols];
    float[] scales = new float[batchSize * (cols / 32)];

    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        expectedFirst,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * firstRows * 8]);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        expectedSecond,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * secondRows * 8]);

    VectorUtil.ggufQ4_0Q8_0DualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        batchSize,
        cols,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * (firstRows + secondRows) * 8],
        GgufQ4Kernel.SHORT_PAIRWISE);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q4_0Q8_0TripleBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 1024;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 1;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow =
        repeat(q4Block(0.125f, ones(32), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF), cols / 32);
    byte[] secondRow =
        repeat(q4Block(-0.25f, ones(32), (lo, hi) -> (lo * 7 + hi) & 0xFF), cols / 32);
    byte[] thirdRow =
        repeat(q4Block(0.0625f, ones(32), (lo, hi) -> (lo * 11 + hi * 13) & 0xFF), cols / 32);
    MemorySegment firstWeight = MemorySegment.ofArray(repeat(firstRow, firstRows));
    MemorySegment secondWeight = MemorySegment.ofArray(repeat(secondRow, secondRows));
    MemorySegment thirdWeight = MemorySegment.ofArray(repeat(thirdRow, thirdRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] expectedThird = new float[batchSize * thirdRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];
    float[] actualThird = new float[batchSize * thirdRows];
    float[] unsignedFirst = new float[batchSize * firstRows];
    float[] unsignedSecond = new float[batchSize * secondRows];
    float[] unsignedThird = new float[batchSize * thirdRows];
    byte[] quants = new byte[batchSize * cols];
    float[] scales = new float[batchSize * (cols / 32)];

    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        expectedFirst,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * firstRows * 8]);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        expectedSecond,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * secondRows * 8]);
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        thirdWeight,
        batchSize,
        thirdRows,
        cols,
        expectedThird,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * thirdRows * 8]);

    VectorUtil.ggufQ4_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        batchSize,
        cols,
        quants,
        scales,
        q4Corrections(batchSize * cols),
        new float[batchSize * (firstRows + secondRows + thirdRows) * 8],
        GgufQ4Kernel.SHORT_PAIRWISE);
    VectorUtil.ggufQ4_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        unsignedFirst,
        secondWeight,
        secondRows,
        unsignedSecond,
        thirdWeight,
        thirdRows,
        unsignedThird,
        batchSize,
        cols,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 32)],
        q4Corrections(batchSize * cols),
        new float[batchSize * (firstRows + secondRows + thirdRows) * 8],
        GgufQ4Kernel.UNSIGNED_PAIRWISE);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
    assertThat(unsignedFirst).containsExactly(expectedFirst);
    assertThat(unsignedSecond).containsExactly(expectedSecond);
    assertThat(unsignedThird).containsExactly(expectedThird);
  }

  @Test
  void q4_0Q8_0SingleQueryBatchMatchesGemvExactly() {
    int rows = 2;
    int cols = 32;
    float[] query = patternedQuery(cols);
    byte[] row0 = q4Block(0.125f, ones(cols), (lo, hi) -> (lo * 5 + hi * 3) & 0xFF);
    byte[] row1 = q4Block(-0.25f, ones(cols), (lo, hi) -> (lo * 7 + hi) & 0xFF);
    float[] expected = new float[rows];
    float[] actual = new float[rows];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));
      VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
          query,
          segment,
          rows,
          cols,
          expected,
          new byte[cols],
          new float[cols / 32],
          q4Corrections(cols));

      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          query,
          segment,
          1,
          rows,
          cols,
          actual,
          new byte[cols],
          new float[cols / 32],
          q4Corrections(cols));

      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q4_0Q8_0BatchedMatmulMatchesGemvReductionAcrossBlocks() {
    int batchSize = 3;
    int rows = 2;
    int cols = 256;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.sin((batch + 1.0) * (col + 0.5)) * (batch + 0.25f);
      }
    }
    byte[] row0 = repeat(q4Block(0.13f, ones(32), (lo, hi) -> (lo * 11 + hi * 7 + 3) & 0xFF), 8);
    byte[] row1 = repeat(q4Block(-0.07f, ones(32), (lo, hi) -> (lo * 5 + hi * 13 + 9) & 0xFF), 8);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));
      for (int batch = 0; batch < batchSize; batch++) {
        float[] query = new float[cols];
        float[] result = new float[rows];
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
            query,
            segment,
            rows,
            cols,
            result,
            new byte[cols],
            new float[cols / 32],
            q4Corrections(cols));
        System.arraycopy(result, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          queries,
          segment,
          batchSize,
          rows,
          cols,
          actual,
          new byte[batchSize * cols],
          new float[batchSize * (cols / 32)],
          q4Corrections(batchSize * cols),
          new float[batchSize * rows * 8]);

      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q4_0Q8_0BatchedMatmulMatchesGemvAtQwenProjectionScaleAfterWarmup() {
    int batchSize = 4;
    int rows = 1024;
    int cols = 1024;
    int blocks = cols / 32;
    Random random = new Random(42L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 8.0f - 4.0f;
    }
    byte[] matrix = new byte[rows * blocks * 18];
    random.nextBytes(matrix);
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += 18) {
      matrixBuffer.putShort(offset, Float.floatToFloat16(0.001f + random.nextFloat() * 0.1f));
    }

    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] gemvOut = new float[rows];
    byte[] gemvQuants = new byte[cols];
    float[] gemvScales = new float[blocks];
    int[] gemvCorrections = q4Corrections(cols);
    byte[] batchQuants = new byte[batchSize * cols];
    float[] batchScales = new float[batchSize * blocks];
    int[] batchCorrections = q4Corrections(batchSize * cols);
    float[] batchLanes = new float[batchSize * rows * 8];

    for (int iteration = 0; iteration < 12; iteration++) {
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
            query, weights, rows, cols, gemvOut, gemvQuants, gemvScales, gemvCorrections);
        System.arraycopy(gemvOut, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
          queries,
          weights,
          batchSize,
          rows,
          cols,
          actual,
          batchQuants,
          batchScales,
          batchCorrections,
          batchLanes);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q4_0Q8_0BatchedMatmulRejectsUndersizedLaneScratch() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, repeat(q4Block(0.1f, ones(32), null), 2));

      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
                      patternedQuery(64),
                      segment,
                      2,
                      1,
                      32,
                      new float[2],
                      new byte[64],
                      new float[2],
                      q4Corrections(64),
                      new float[15]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("lane scratch");
    }
  }

  @Test
  void q8_0BatchDotProduct_respectsRowOffsets() {
    float[] query = ones(32);
    byte[] row0 = q8Block(1.0f);
    byte[] row1 = q8Block(-0.5f);
    byte[] matrix = concat(row0, row1);
    float[] out = new float[2];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, matrix);

      VectorUtil.ggufQ8_0BatchDotProduct(query, segment, 2, 32, out);

      assertThat(out[0])
          .isCloseTo(VectorUtil.ggufQ8_0DotProduct(query, segment, 0, 32), within(1e-5f));
      assertThat(out[1])
          .isCloseTo(VectorUtil.ggufQ8_0DotProduct(query, segment, 34, 32), within(1e-5f));
    }
  }

  @Test
  void q8_0Q8_0BatchDotProduct_quantizesEachQueryBlockUsingGgmlSemantics() {
    float[] query = new float[32];
    query[0] = 1.0f;
    query[1] = 0.49f;
    byte[] row0 = q8Block(1.0f, ignored -> 1);
    byte[] row1 = q8Block(1.0f, ignored -> 2);
    float[] out = new float[2];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[query.length / 32];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
          query, segment, 2, query.length, out, q8Quants, q8Scales);

      float q8Scale = Float.float16ToFloat(Float.floatToFloat16(1.0f / 127.0f));
      assertThat(out[0]).isCloseTo(189.0f * q8Scale, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f * q8Scale, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
    }
  }

  @Test
  void q8_0Q8_0DualBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 64;
    int firstRows = 2;
    int secondRows = 3;
    float[] query = patternedQuery(cols);
    MemorySegment firstWeight =
        MemorySegment.ofArray(repeat(q8Block(0.125f, i -> i * 7 - 53), firstRows * 2));
    MemorySegment secondWeight =
        MemorySegment.ofArray(repeat(q8Block(-0.25f, i -> 61 - i * 5), secondRows * 2));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];

    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
        query, firstWeight, firstRows, cols, expectedFirst, new byte[cols], new float[cols / 32]);
    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 32]);

    VectorUtil.ggufQ8_0Q8_0DualBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        cols,
        new byte[cols],
        new float[cols / 32]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q8_0Q8_0TripleBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 64;
    int firstRows = 4;
    int secondRows = 2;
    int thirdRows = 3;
    float[] query = patternedQuery(cols);
    MemorySegment firstWeight =
        MemorySegment.ofArray(repeat(q8Block(0.125f, i -> i * 7 - 53), firstRows * 2));
    MemorySegment secondWeight =
        MemorySegment.ofArray(repeat(q8Block(-0.25f, i -> 61 - i * 5), secondRows * 2));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(repeat(q8Block(0.03125f, i -> i * 3 - 41), thirdRows * 2));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] expectedThird = new float[thirdRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];
    float[] actualThird = new float[thirdRows];

    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
        query, firstWeight, firstRows, cols, expectedFirst, new byte[cols], new float[cols / 32]);
    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 32]);
    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
        query, thirdWeight, thirdRows, cols, expectedThird, new byte[cols], new float[cols / 32]);

    VectorUtil.ggufQ8_0Q8_0TripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        cols,
        new byte[cols],
        new float[cols / 32]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q8_0Q8_0BatchedMatmulMatchesIndependentQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 64;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.sin((batch + 0.375) * (col + 0.625)) * (batch + 0.75f);
      }
    }
    byte[] firstBlock = q8Block(0.125f, index -> index * 7 - 53);
    byte[] secondBlock = q8Block(-0.25f, index -> 61 - index * 5);
    MemorySegment weights =
        MemorySegment.ofArray(
            concat(concat(firstBlock, secondBlock), concat(secondBlock, firstBlock)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];

    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
          query, weights, rows, cols, result, new byte[cols], new float[cols / 32]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        actual,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 32)]);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0PrequantizedRowRangesComposeToTheExactBatchedResult() {
    int batchSize = 3;
    int rows = 5;
    int cols = 64;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow = repeat(q8Block(0.125f, index -> index * 7 - 53), 2);
    byte[] secondRow = repeat(q8Block(-0.25f, index -> 61 - index * 5), 2);
    MemorySegment weights =
        MemorySegment.ofArray(
            concat(concat(firstRow, secondRow), concat(concat(firstRow, secondRow), firstRow)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(batchSize, cols);

    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        expected,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 32)]);
    activation.quantize(queries, batchSize);

    VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
        weights, batchSize, rows, cols, 0, 2, actual, activation);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
        weights, batchSize, rows, cols, 2, rows, actual, activation);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0BlockRangeActivationQuantizationMatchesWholeBatchExactly() {
    int batchSize = 3;
    int rows = 5;
    int cols = 64;
    int blocks = cols / 32;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] row = repeat(q8Block(0.125f, index -> index * 11 - 47), blocks);
    MemorySegment weights = MemorySegment.ofArray(repeat(row, rows));
    GgufQ8_0Batch whole = GgufQ8_0Batch.allocate(batchSize, cols);
    GgufQ8_0Batch rangeWise = GgufQ8_0Batch.allocate(batchSize, cols);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    whole.quantize(queries, batchSize);
    rangeWise.quantizeBlockRange(queries, batchSize, 0, 1);
    rangeWise.quantizeBlockRange(queries, batchSize, 1, blocks);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
        weights, batchSize, rows, cols, 0, rows, expected, whole);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
        weights, batchSize, rows, cols, 0, rows, actual, rangeWise);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0Q8_0DualBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 64;
    int firstRows = 2;
    int secondRows = 3;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow = repeat(q8Block(0.125f, index -> index * 7 - 53), 2);
    byte[] secondRow = repeat(q8Block(-0.25f, index -> 61 - index * 5), 2);
    MemorySegment firstWeight = MemorySegment.ofArray(repeat(firstRow, firstRows));
    MemorySegment secondWeight = MemorySegment.ofArray(repeat(secondRow, secondRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];
    byte[] quants = new byte[batchSize * cols];
    float[] scales = new float[batchSize * (cols / 32)];

    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, firstWeight, batchSize, firstRows, cols, expectedFirst, quants, scales);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, secondWeight, batchSize, secondRows, cols, expectedSecond, quants, scales);

    VectorUtil.ggufQ8_0Q8_0DualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        batchSize,
        cols,
        quants,
        scales);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q8_0Q8_0TripleBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 64;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 1;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow = repeat(q8Block(0.125f, index -> index * 7 - 53), 2);
    byte[] secondRow = repeat(q8Block(-0.25f, index -> 61 - index * 5), 2);
    byte[] thirdRow = repeat(q8Block(0.03125f, index -> index * 3 - 41), 2);
    MemorySegment firstWeight = MemorySegment.ofArray(repeat(firstRow, firstRows));
    MemorySegment secondWeight = MemorySegment.ofArray(repeat(secondRow, secondRows));
    MemorySegment thirdWeight = MemorySegment.ofArray(repeat(thirdRow, thirdRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] expectedThird = new float[batchSize * thirdRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];
    float[] actualThird = new float[batchSize * thirdRows];
    byte[] quants = new byte[batchSize * cols];
    float[] scales = new float[batchSize * (cols / 32)];

    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, firstWeight, batchSize, firstRows, cols, expectedFirst, quants, scales);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, secondWeight, batchSize, secondRows, cols, expectedSecond, quants, scales);
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, thirdWeight, batchSize, thirdRows, cols, expectedThird, quants, scales);

    VectorUtil.ggufQ8_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        batchSize,
        cols,
        quants,
        scales);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q8_0Q8_0BatchedMatmulMatchesGemvAtProjectionScaleAfterWarmup() {
    int batchSize = 4;
    int rows = 512;
    int cols = 2_048;
    int blocks = cols / 32;
    int blockBytes = 34;
    Random random = new Random(0xB47C_8B48L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 8.0f - 4.0f;
    }
    byte[] matrix = new byte[rows * blocks * blockBytes];
    random.nextBytes(matrix);
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += blockBytes) {
      matrixBuffer.putShort(offset, Float.floatToFloat16(0.001f + random.nextFloat() * 0.1f));
    }

    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] gemvOut = new float[rows];
    byte[] gemvQuants = new byte[cols];
    float[] gemvScales = new float[blocks];
    byte[] batchQuants = new byte[batchSize * cols];
    float[] batchScales = new float[batchSize * blocks];

    for (int iteration = 0; iteration < 12; iteration++) {
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
            query, weights, rows, cols, gemvOut, gemvQuants, gemvScales);
        System.arraycopy(gemvOut, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
          queries, weights, batchSize, rows, cols, actual, batchQuants, batchScales);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q8_0Q8_0BatchedMatmulRejectsUndersizedScaleScratch() {
    int batchSize = 2;
    int cols = 32;
    MemorySegment weights = MemorySegment.ofArray(q8Block(1.0f));

    assertThatThrownBy(
            () ->
                VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
                    new float[batchSize * cols],
                    weights,
                    batchSize,
                    1,
                    cols,
                    new float[batchSize],
                    new byte[batchSize * cols],
                    new float[batchSize - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("q8Scales");
  }

  @Test
  void activationQuantizedBatchDotProducts_rejectUndersizedScratch() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q4 = copy(arena, q4Block(1.0f, ones(32), null));
      MemorySegment q4K =
          copy(
              arena,
              q4KBlock(1.0f, 0.0f, ignored -> 1, new int[] {1, 1, 1, 1, 1, 1, 1, 1}, new int[8]));
      MemorySegment q8 = copy(arena, q8Block(1.0f));
      MemorySegment q5K =
          copy(
              arena,
              q5KBlock(1.0f, 0.0f, ignored -> 1, new int[] {1, 1, 1, 1, 1, 1, 1, 1}, new int[8]));
      MemorySegment q6 = copy(arena, q6KBlock(1.0f, ignored -> 1, ignored -> 1));

      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
                      ones(32),
                      q4,
                      1,
                      32,
                      new float[1],
                      new byte[31],
                      new float[1],
                      q4Corrections(32)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Quants.length");
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
                      ones(32), q4, 1, 32, new float[1], new byte[32], new float[1], new int[7]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8ZeroPointCorrections.length");
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
                      ones(32), q8, 1, 32, new float[1], new byte[32], new float[0]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Scales.length");
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
                      ones(256), q6, 1, 256, new float[1], new byte[255], new float[1]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Quants.length");
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
                      ones(256),
                      q4K,
                      1,
                      256,
                      new float[1],
                      new byte[256],
                      new float[1],
                      new short[15]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Sums.length");
      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
                      ones(256),
                      q5K,
                      1,
                      256,
                      new float[1],
                      new byte[256],
                      new float[1],
                      new short[15]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Sums.length");
    }
  }

  @Test
  void q6_KBatchDotProduct_respectsRowOffsets() {
    float[] query = patternedQuery(256);
    byte[] row0 = q6KBlock(0.125f, i -> (i % 64) - 32, i -> (i % 7) - 3);
    byte[] row1 = q6KBlock(-0.25f, i -> 31 - (i % 64), i -> (i % 5) - 2);
    byte[] matrix = concat(row0, row1);
    float[] out = new float[2];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, matrix);

      VectorUtil.ggufQ6_KBatchDotProduct(query, segment, 2, 256, out);

      assertThat(out[0])
          .isCloseTo(VectorUtil.ggufQ6_KDotProduct(query, segment, 0, 256), within(1e-5f));
      assertThat(out[1])
          .isCloseTo(VectorUtil.ggufQ6_KDotProduct(query, segment, 210, 256), within(1e-5f));
    }
  }

  @Test
  void q6_KQ8_KBatchDotProduct_quantizesTheQueryOnceUsingGgmlSemantics() {
    float[] query = new float[256];
    query[0] = 1.0f;
    query[1] = 0.49f;
    byte[] row0 = q6KBlock(1.0f, ignored -> 1, ignored -> 1);
    byte[] row1 = q6KBlock(1.0f, ignored -> 2, ignored -> 1);
    float[] out = new float[2];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[query.length / 256];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
          query, segment, 2, query.length, out, q8Quants, q8Scales);

      assertThat(out[0]).isCloseTo(189.0f / 127.0f, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f / 127.0f, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
    }
  }

  @Test
  void q6_KQ8_KBatchDotProduct_reusesIndependentScratchAcrossParallelRows() {
    int rows = 4096;
    float[] query = patternedQuery(256);
    byte[] row = q6KBlock(0.125f, index -> (index * 5 + 3) % 64 - 32, index -> index - 8);
    float[] expected = new float[1];
    float[] out = new float[rows];
    byte[] expectedQuants = new byte[query.length];
    float[] expectedScales = new float[1];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[1];

    new ScalarVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query,
            MemorySegment.ofArray(row),
            1,
            query.length,
            expected,
            expectedQuants,
            expectedScales);
    VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
        query,
        MemorySegment.ofArray(repeat(row, rows)),
        rows,
        query.length,
        out,
        q8Quants,
        q8Scales);

    assertThat(q8Quants).containsExactly(expectedQuants);
    assertThat(q8Scales).containsExactly(expectedScales);
    for (float value : out) {
      assertThat(value).isCloseTo(expected[0], within(1e-4f));
    }
  }

  @Test
  void q6_KQ8_KBatchedMatmulMatchesIndependentQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.cos((batch + 0.75) * (col + 0.5)) * (batch + 0.25f);
      }
    }
    byte[] firstBlock = q6KBlock(0.125f, index -> (index * 5 + 3) % 64 - 32, index -> index - 8);
    byte[] secondBlock = q6KBlock(-0.25f, index -> 31 - (index % 64), index -> 7 - index);
    MemorySegment weights =
        MemorySegment.ofArray(
            concat(concat(firstBlock, secondBlock), concat(secondBlock, firstBlock)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];

    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
          query, weights, rows, cols, result, new byte[cols], new float[cols / 256]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        actual,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)]);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q6_KQ8_KBatchedMatmulMatchesGemvAtProjectionScaleAfterWarmup() {
    int batchSize = 4;
    int rows = 512;
    int cols = 2_048;
    int blocks = cols / 256;
    int blockBytes = 210;
    Random random = new Random(0xB47C_6B48L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 8.0f - 4.0f;
    }
    byte[] matrix = new byte[rows * blocks * blockBytes];
    random.nextBytes(matrix);
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += blockBytes) {
      matrixBuffer.putShort(offset + 208, Float.floatToFloat16(0.001f + random.nextFloat() * 0.1f));
    }

    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] gemvOut = new float[rows];
    byte[] gemvQuants = new byte[cols];
    float[] gemvScales = new float[blocks];
    byte[] batchQuants = new byte[batchSize * cols];
    float[] batchScales = new float[batchSize * blocks];

    for (int iteration = 0; iteration < 12; iteration++) {
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
            query, weights, rows, cols, gemvOut, gemvQuants, gemvScales);
        System.arraycopy(gemvOut, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
          queries, weights, batchSize, rows, cols, actual, batchQuants, batchScales);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q6_KQ8_KBatchedMatmulRejectsUndersizedScaleScratch() {
    int batchSize = 2;
    int cols = 256;
    MemorySegment weights = MemorySegment.ofArray(q6KBlock(1.0f, ignored -> 1, ignored -> 1));

    assertThatThrownBy(
            () ->
                VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
                    new float[batchSize * cols],
                    weights,
                    batchSize,
                    1,
                    cols,
                    new float[batchSize],
                    new byte[batchSize * cols],
                    new float[batchSize - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("q8Scales");
  }

  @Test
  void q4_KQ8_KBatchDotProduct_quantizesTheQueryOnceUsingGgmlSemantics() {
    float[] query = new float[256];
    query[0] = 1.0f;
    query[1] = 0.49f;
    int[] unitScales = {1, 1, 1, 1, 1, 1, 1, 1};
    int[] zeroMins = new int[8];
    byte[] row0 = q4KBlock(1.0f, 0.0f, ignored -> 1, unitScales, zeroMins);
    byte[] row1 = q4KBlock(1.0f, 0.0f, ignored -> 2, unitScales, zeroMins);
    float[] out = new float[2];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[query.length / 256];
    short[] q8Sums = new short[query.length / 16];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
          query, segment, 2, query.length, out, q8Quants, q8Scales, q8Sums);

      assertThat(out[0]).isCloseTo(189.0f / 127.0f, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f / 127.0f, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
    }
  }

  @Test
  void q4_KQ8_KBatchDotProduct_matchesDequantizedReferenceWithMins() {
    float[] query = patternedQuery(256);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] row = q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins);
    float[] out = new float[1];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[1];
    short[] q8Sums = new short[16];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, row);

      VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
          query, segment, 1, query.length, out, q8Quants, q8Scales, q8Sums);
    }

    float expected = 0.0f;
    for (int index = 0; index < query.length; index++) {
      int group = index / 32;
      float weight = 0.125f * scales[group] * (index % 16) - 0.0625f * mins[group];
      float activation = q8Scales[0] * q8Quants[index];
      expected = Math.fma(weight, activation, expected);
    }
    assertThat(out[0]).isCloseTo(expected, within(1e-3f));
  }

  @Test
  void q4_KQ8_KBatchedMatmulMatchesIndependentQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.sin((batch + 1.0) * (col + 0.25)) * (batch + 0.5f);
      }
    }
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstBlock = q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins);
    byte[] secondBlock = q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins);
    MemorySegment weights =
        MemorySegment.ofArray(
            concat(concat(firstBlock, secondBlock), concat(secondBlock, firstBlock)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];

    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
          query,
          weights,
          rows,
          cols,
          result,
          new byte[cols],
          new float[cols / 256],
          new short[cols / 16]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        actual,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)],
        new short[batchSize * (cols / 16)]);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q4_KQ8_KBatchedMatmulMatchesGemvAtProjectionScaleAfterWarmup() {
    int batchSize = 4;
    int rows = 512;
    int cols = 2_048;
    int blocks = cols / 256;
    Random random = new Random(0xB47C_4B48L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 8.0f - 4.0f;
    }
    byte[] matrix = new byte[rows * blocks * 144];
    random.nextBytes(matrix);
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += 144) {
      matrixBuffer.putShort(offset, Float.floatToFloat16(0.001f + random.nextFloat() * 0.1f));
      matrixBuffer.putShort(offset + Short.BYTES, Float.floatToFloat16(random.nextFloat() * 0.05f));
    }

    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] gemvOut = new float[rows];
    byte[] gemvQuants = new byte[cols];
    float[] gemvScales = new float[blocks];
    short[] gemvSums = new short[cols / 16];
    byte[] batchQuants = new byte[batchSize * cols];
    float[] batchScales = new float[batchSize * blocks];
    short[] batchSums = new short[batchSize * (cols / 16)];

    for (int iteration = 0; iteration < 12; iteration++) {
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
            query, weights, rows, cols, gemvOut, gemvQuants, gemvScales, gemvSums);
        System.arraycopy(gemvOut, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
          queries, weights, batchSize, rows, cols, actual, batchQuants, batchScales, batchSums);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q4_KQ8_KBatchedMatmulRejectsUndersizedSumScratch() {
    int batchSize = 2;
    int cols = 256;
    int[] unitScales = {1, 1, 1, 1, 1, 1, 1, 1};
    MemorySegment weights =
        MemorySegment.ofArray(q4KBlock(1.0f, 0.0f, ignored -> 1, unitScales, new int[8]));

    assertThatThrownBy(
            () ->
                VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
                    new float[batchSize * cols],
                    weights,
                    batchSize,
                    1,
                    cols,
                    new float[batchSize],
                    new byte[batchSize * cols],
                    new float[batchSize],
                    new short[batchSize * (cols / 16) - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("q8Sums");
  }

  @Test
  void q4_KQ8_KDualBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 512;
    int firstRows = 2;
    int secondRows = 3;
    float[] queries = patternedQueries(batchSize, cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstRow =
        concat(
            q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins),
            q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins));
    byte[] secondRow =
        concat(
            q4KBlock(0.5f, -0.015625f, i -> i * 3 % 16, scales, mins),
            q4KBlock(0.0625f, 0.125f, i -> i * 5 % 16, scales, mins));
    MemorySegment firstWeight = MemorySegment.ofArray(repeat(firstRow, firstRows));
    MemorySegment secondWeight = MemorySegment.ofArray(repeat(secondRow, secondRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];

    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        expectedFirst,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)],
        new short[batchSize * (cols / 16)]);
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        expectedSecond,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)],
        new short[batchSize * (cols / 16)]);

    VectorUtil.ggufQ4_KQ8_KDualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        batchSize,
        cols,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)],
        new short[batchSize * (cols / 16)]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q4_KQ8_KDualBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 256;
    int firstRows = 3;
    int secondRows = 2;
    float[] query = patternedQuery(cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins), secondRows));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];

    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    VectorUtil.ggufQ4_KQ8_KDualBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q4_KQ8_KTripleBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 256;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 1;
    float[] query = patternedQuery(cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins), secondRows));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(0.5f, -0.015625f, i -> i * 3 % 16, scales, mins), thirdRows));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] expectedThird = new float[thirdRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];
    float[] actualThird = new float[thirdRows];

    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        thirdWeight,
        thirdRows,
        cols,
        expectedThird,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    VectorUtil.ggufQ4_KQ8_KTripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q4_KQ4_KQ6_KQ8_KTripleBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 256;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 2;
    float[] query = patternedQuery(cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins), secondRows));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(
            repeat(q6KBlock(0.25f, i -> (i * 5 + 3) % 64 - 32, i -> i - 8), thirdRows));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] expectedThird = new float[thirdRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];
    float[] actualThird = new float[thirdRows];

    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
        query, thirdWeight, thirdRows, cols, expectedThird, new byte[cols], new float[cols / 256]);

    VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q4_KQ4_KQ6_KQ8_KTripleBatchedMatmulMatchesSeparateBatchedMatmulsExactly() {
    int batchSize = 3;
    int cols = 256;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 2;
    float[] queries = patternedQueries(batchSize, cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(0.125f, 0.0625f, i -> i % 16, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q4KBlock(-0.25f, 0.03125f, i -> 15 - i % 16, scales, mins), secondRows));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(
            repeat(q6KBlock(0.25f, i -> (i * 5 + 3) % 64 - 32, i -> i - 8), thirdRows));
    float[] expectedFirst = new float[batchSize * firstRows];
    float[] expectedSecond = new float[batchSize * secondRows];
    float[] expectedThird = new float[batchSize * thirdRows];
    float[] actualFirst = new float[batchSize * firstRows];
    float[] actualSecond = new float[batchSize * secondRows];
    float[] actualThird = new float[batchSize * thirdRows];
    byte[] quants = new byte[batchSize * cols];
    float[] quantScales = new float[batchSize * (cols / 256)];
    short[] quantSums = new short[batchSize * (cols / 16)];

    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        expectedFirst,
        quants,
        quantScales,
        quantSums);
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        expectedSecond,
        quants,
        quantScales,
        quantSums);
    VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
        queries, thirdWeight, batchSize, thirdRows, cols, expectedThird, quants, quantScales);

    VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        batchSize,
        cols,
        quants,
        quantScales,
        quantSums);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q5_KQ8_KDualBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 256;
    int firstRows = 3;
    int secondRows = 2;
    float[] query = patternedQuery(cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q5KBlock(-0.25f, 0.03125f, i -> 31 - i % 32, scales, mins), secondRows));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];

    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    VectorUtil.ggufQ5_KQ8_KDualBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
  }

  @Test
  void q5_KQ8_KTripleBatchDotProductMatchesSeparateMatmulsExactly() {
    int cols = 256;
    int firstRows = 2;
    int secondRows = 3;
    int thirdRows = 1;
    float[] query = patternedQuery(cols);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    MemorySegment firstWeight =
        MemorySegment.ofArray(
            repeat(q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins), firstRows));
    MemorySegment secondWeight =
        MemorySegment.ofArray(
            repeat(q5KBlock(-0.25f, 0.03125f, i -> 31 - i % 32, scales, mins), secondRows));
    MemorySegment thirdWeight =
        MemorySegment.ofArray(
            repeat(q5KBlock(0.5f, -0.015625f, i -> i * 11 % 32, scales, mins), thirdRows));
    float[] expectedFirst = new float[firstRows];
    float[] expectedSecond = new float[secondRows];
    float[] expectedThird = new float[thirdRows];
    float[] actualFirst = new float[firstRows];
    float[] actualSecond = new float[secondRows];
    float[] actualThird = new float[thirdRows];

    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        cols,
        expectedFirst,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query,
        secondWeight,
        secondRows,
        cols,
        expectedSecond,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);
    VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
        query,
        thirdWeight,
        thirdRows,
        cols,
        expectedThird,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    VectorUtil.ggufQ5_KQ8_KTripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        actualFirst,
        secondWeight,
        secondRows,
        actualSecond,
        thirdWeight,
        thirdRows,
        actualThird,
        cols,
        new byte[cols],
        new float[cols / 256],
        new short[cols / 16]);

    assertThat(actualFirst).containsExactly(expectedFirst);
    assertThat(actualSecond).containsExactly(expectedSecond);
    assertThat(actualThird).containsExactly(expectedThird);
  }

  @Test
  void q5_KQ8_KBatchedMatmulMatchesIndependentQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.sin((batch + 0.5) * (col + 0.75)) * (batch + 0.625f);
      }
    }
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] firstBlock = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);
    byte[] secondBlock = q5KBlock(-0.25f, 0.03125f, i -> 31 - i % 32, scales, mins);
    MemorySegment weights =
        MemorySegment.ofArray(
            concat(concat(firstBlock, secondBlock), concat(secondBlock, firstBlock)));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];

    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
          query,
          weights,
          rows,
          cols,
          result,
          new byte[cols],
          new float[cols / 256],
          new short[cols / 16]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    VectorUtil.ggufQ5_KQ8_KBatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        actual,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 256)],
        new short[batchSize * (cols / 16)]);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q5_KQ8_KBatchedMatmulMatchesGemvAtProjectionScaleAfterWarmup() {
    int batchSize = 4;
    int rows = 512;
    int cols = 2_048;
    int blocks = cols / 256;
    int blockBytes = 176;
    Random random = new Random(0xB47C_5B48L);
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 8.0f - 4.0f;
    }
    byte[] matrix = new byte[rows * blocks * blockBytes];
    random.nextBytes(matrix);
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += blockBytes) {
      matrixBuffer.putShort(offset, Float.floatToFloat16(0.001f + random.nextFloat() * 0.1f));
      matrixBuffer.putShort(offset + Short.BYTES, Float.floatToFloat16(random.nextFloat() * 0.05f));
    }

    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] gemvOut = new float[rows];
    byte[] gemvQuants = new byte[cols];
    float[] gemvScales = new float[blocks];
    short[] gemvSums = new short[cols / 16];
    byte[] batchQuants = new byte[batchSize * cols];
    float[] batchScales = new float[batchSize * blocks];
    short[] batchSums = new short[batchSize * (cols / 16)];

    for (int iteration = 0; iteration < 12; iteration++) {
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(queries, batch * cols, query, 0, cols);
        VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
            query, weights, rows, cols, gemvOut, gemvQuants, gemvScales, gemvSums);
        System.arraycopy(gemvOut, 0, expected, batch * rows, rows);
      }

      VectorUtil.ggufQ5_KQ8_KBatchedMatmul(
          queries, weights, batchSize, rows, cols, actual, batchQuants, batchScales, batchSums);
      assertThat(actual).containsExactly(expected);
    }
  }

  @Test
  void q5_KQ8_KBatchedMatmulRejectsUndersizedSumScratch() {
    int batchSize = 2;
    int cols = 256;
    int[] unitScales = {1, 1, 1, 1, 1, 1, 1, 1};
    MemorySegment weights =
        MemorySegment.ofArray(q5KBlock(1.0f, 0.0f, ignored -> 1, unitScales, new int[8]));

    assertThatThrownBy(
            () ->
                VectorUtil.ggufQ5_KQ8_KBatchedMatmul(
                    new float[batchSize * cols],
                    weights,
                    batchSize,
                    1,
                    cols,
                    new float[batchSize],
                    new byte[batchSize * cols],
                    new float[batchSize],
                    new short[batchSize * (cols / 16) - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("q8Sums");
  }

  @Test
  void q5_KBatchDotProduct_respectsRowOffsets() {
    float[] query = patternedQuery(256);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] row0 = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);
    byte[] row1 = q5KBlock(-0.25f, 0.03125f, i -> 31 - (i % 32), scales, mins);
    float[] out = new float[2];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ5_KBatchDotProduct(query, segment, 2, query.length, out);

      assertThat(out[0])
          .isCloseTo(VectorUtil.ggufQ5_KDotProduct(query, segment, 0, 256), within(1e-5f));
      assertThat(out[1])
          .isCloseTo(VectorUtil.ggufQ5_KDotProduct(query, segment, 176, 256), within(1e-5f));
    }
  }

  @Test
  void q5_KQ8_KBatchDotProduct_matchesQuantizedActivationReferenceWithMins() {
    float[] query = patternedQuery(256);
    int[] scales = {5, 12, 30, 60, 7, 15, 31, 63};
    int[] mins = {3, 8, 20, 45, 1, 10, 25, 50};
    byte[] row = q5KBlock(0.125f, 0.0625f, i -> (i * 7 + 3) % 32, scales, mins);
    float[] out = new float[1];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[1];
    short[] q8Sums = new short[16];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, row);

      VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
          query, segment, 1, query.length, out, q8Quants, q8Scales, q8Sums);
    }

    float expected = 0.0f;
    for (int index = 0; index < query.length; index++) {
      int group = index / 32;
      int quant = (index * 7 + 3) % 32;
      float weight = 0.125f * scales[group] * quant - 0.0625f * mins[group];
      float activation = q8Scales[0] * q8Quants[index];
      expected = Math.fma(weight, activation, expected);
    }
    assertThat(out[0]).isCloseTo(expected, within(1e-3f));
  }

  @Test
  void q5_0Q8_0BatchDotProduct_quantizesTheQueryOnceUsingGgmlSemantics() {
    float[] query = new float[32];
    query[0] = 1.0f;
    query[1] = 0.49f;
    byte[] row0 = q5Block(1.0f, ignored -> 1);
    byte[] row1 = q5Block(1.0f, ignored -> 2);
    float[] out = new float[2];
    byte[] q8Quants = new byte[query.length];
    float[] q8Scales = new float[query.length / 32];

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, concat(row0, row1));

      VectorUtil.ggufQ5_0Q8_0BatchDotProduct(
          query, segment, 2, query.length, out, q8Quants, q8Scales);

      float q8Scale = Float.float16ToFloat(Float.floatToFloat16(1.0f / 127.0f));
      assertThat(out[0]).isCloseTo(189.0f * q8Scale, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f * q8Scale, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
    }
  }

  @Test
  void q5_0Q8_0BatchedMatmulMatchesIndependentQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 64;
    float[] queries = patternedQueries(batchSize, cols);
    byte[] firstRow =
        concat(
            q5Block(0.125f, index -> (index * 7) % 32 - 16),
            q5Block(-0.25f, index -> 15 - (index * 5) % 32));
    byte[] secondRow =
        concat(
            q5Block(0.0625f, index -> (index * 11) % 32 - 16),
            q5Block(0.5f, index -> 15 - (index * 3) % 32));
    MemorySegment weights = MemorySegment.ofArray(concat(firstRow, secondRow));
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];

    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      float[] result = new float[rows];
      VectorUtil.ggufQ5_0Q8_0BatchDotProduct(
          query, weights, rows, cols, result, new byte[cols], new float[cols / 32]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    VectorUtil.ggufQ5_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        actual,
        new byte[batchSize * cols],
        new float[batchSize * (cols / 32)]);

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q5_0Q8_0BatchedMatmulRejectsUndersizedScaleScratch() {
    int batchSize = 2;
    int cols = 32;
    MemorySegment weights = MemorySegment.ofArray(q5Block(1.0f, ignored -> 1));

    assertThatThrownBy(
            () ->
                VectorUtil.ggufQ5_0Q8_0BatchedMatmul(
                    new float[batchSize * cols],
                    weights,
                    batchSize,
                    1,
                    cols,
                    new float[batchSize],
                    new byte[batchSize * cols],
                    new float[batchSize - 1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("q8Scales");
  }

  @Test
  void quantizedDotRejectsNonBlockAlignedDimensions() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q4 = copy(arena, q4Block(1.0f, ones(32), null));
      MemorySegment q4K =
          copy(
              arena,
              q4KBlock(1.0f, 0.0f, ignored -> 1, new int[] {1, 1, 1, 1, 1, 1, 1, 1}, new int[8]));
      MemorySegment q8 = copy(arena, q8Block(1.0f));
      MemorySegment q5 = copy(arena, q5Block(1.0f, ignored -> 1));
      MemorySegment q5K =
          copy(
              arena,
              q5KBlock(1.0f, 0.0f, ignored -> 1, new int[] {1, 1, 1, 1, 1, 1, 1, 1}, new int[8]));
      MemorySegment q6 = copy(arena, q6KBlock(1.0f, i -> 0, i -> 1));

      assertThatThrownBy(() -> VectorUtil.ggufQ4_0DotProduct(ones(31), q4, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ8_0DotProduct(ones(31), q8, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ5_0DotProduct(ones(31), q5, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ6_KDotProduct(ones(255), q6, 0, 255))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 256");
      assertThatThrownBy(() -> VectorUtil.ggufQ4_KDotProduct(ones(255), q4K, 0, 255))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 256");
      assertThatThrownBy(() -> VectorUtil.ggufQ5_KDotProduct(ones(255), q5K, 0, 255))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 256");
    }
  }

  private static byte[] q4Block(float scale, float[] ignored, PackedByteFactory factory) {
    byte[] block = new byte[18];
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, Float.floatToFloat16(scale));
    for (int i = 0; i < 16; i++) {
      if (factory != null) {
        block[2 + i] = (byte) factory.packed(i, i + 16);
      } else {
        int lo = i & 0x0F;
        int hi = 15 - i;
        block[2 + i] = (byte) (lo | (hi << 4));
      }
    }
    return block;
  }

  private static byte[] repeat(byte[] row, int count) {
    byte[] result = new byte[Math.multiplyExact(row.length, count)];
    for (int index = 0; index < count; index++) {
      System.arraycopy(row, 0, result, index * row.length, row.length);
    }
    return result;
  }

  private static byte[] q8Block(float scale) {
    return q8Block(scale, i -> i - 16);
  }

  private static byte[] q8Block(float scale, IntUnaryOperator quantFactory) {
    byte[] block = new byte[34];
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, Float.floatToFloat16(scale));
    for (int i = 0; i < 32; i++) {
      block[2 + i] = (byte) quantFactory.applyAsInt(i);
    }
    return block;
  }

  private static byte[] q5Block(float scale, IntUnaryOperator quantFactory) {
    byte[] block = new byte[22];
    ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    int highBits = 0;
    for (int index = 0; index < 16; index++) {
      int lowQuant = quantFactory.applyAsInt(index) + 16;
      int highQuant = quantFactory.applyAsInt(index + 16) + 16;
      block[6 + index] = (byte) ((lowQuant & 0x0F) | ((highQuant & 0x0F) << 4));
      highBits |= ((lowQuant >>> 4) & 1) << index;
      highBits |= ((highQuant >>> 4) & 1) << (index + 16);
    }
    buffer.putInt(2, highBits);
    return block;
  }

  private static byte[] f32Bytes(float[] values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return buffer.array();
  }

  private static byte[] q6KBlock(
      float scale, IntUnaryOperator quantFactory, IntUnaryOperator scaleFactory) {
    byte[] block = new byte[210];
    for (int i = 0; i < 16; i++) {
      block[192 + i] = (byte) scaleFactory.applyAsInt(i);
    }
    ByteBuffer.wrap(block)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putShort(208, Float.floatToFloat16(scale));

    for (int superBlock = 0; superBlock < 2; superBlock++) {
      int positionBase = superBlock * 128;
      int qlBase = superBlock * 64;
      int qhBase = 128 + superBlock * 32;
      for (int l = 0; l < 32; l++) {
        int q1 = quantFactory.applyAsInt(positionBase + l) + 32;
        int q2 = quantFactory.applyAsInt(positionBase + l + 32) + 32;
        int q3 = quantFactory.applyAsInt(positionBase + l + 64) + 32;
        int q4 = quantFactory.applyAsInt(positionBase + l + 96) + 32;
        block[qlBase + l] = (byte) ((q1 & 0x0F) | ((q3 & 0x0F) << 4));
        block[qlBase + 32 + l] = (byte) ((q2 & 0x0F) | ((q4 & 0x0F) << 4));
        block[qhBase + l] =
            (byte)
                (((q1 >>> 4) & 0x03)
                    | (((q2 >>> 4) & 0x03) << 2)
                    | (((q3 >>> 4) & 0x03) << 4)
                    | (((q4 >>> 4) & 0x03) << 6));
      }
    }
    return block;
  }

  private static byte[] q5KBlock(
      float scale, float minScale, IntUnaryOperator quantFactory, int[] scales, int[] mins) {
    byte[] block = new byte[176];
    ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    buffer.putShort(2, Float.floatToFloat16(minScale));

    for (int group = 0; group < 4; group++) {
      block[4 + group] = (byte) scales[group];
      block[8 + group] = (byte) mins[group];
    }
    for (int group = 4; group < 8; group++) {
      block[4 + group + 4] = (byte) ((scales[group] & 0x0F) | ((mins[group] & 0x0F) << 4));
      block[4 + group - 4] |= (byte) ((scales[group] >>> 4) << 6);
      block[4 + group] |= (byte) ((mins[group] >>> 4) << 6);
    }

    for (int group = 0; group < 8; group++) {
      int byteOffset = 48 + (group >>> 1) * 32;
      int shift = (group & 1) * 4;
      for (int index = 0; index < 32; index++) {
        int quant = quantFactory.applyAsInt(group * 32 + index);
        block[byteOffset + index] |= (byte) ((quant & 0x0F) << shift);
        block[16 + index] |= (byte) (((quant >>> 4) & 1) << group);
      }
    }
    return block;
  }

  private static byte[] q4KBlock(
      float scale, float minScale, IntUnaryOperator quantFactory, int[] scales, int[] mins) {
    byte[] block = new byte[144];
    ByteBuffer buffer = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(0, Float.floatToFloat16(scale));
    buffer.putShort(2, Float.floatToFloat16(minScale));

    for (int group = 0; group < 4; group++) {
      block[4 + group] = (byte) scales[group];
      block[8 + group] = (byte) mins[group];
    }
    for (int group = 4; group < 8; group++) {
      block[4 + group + 4] = (byte) ((scales[group] & 0x0F) | ((mins[group] & 0x0F) << 4));
      block[4 + group - 4] |= (byte) ((scales[group] >>> 4) << 6);
      block[4 + group] |= (byte) ((mins[group] >>> 4) << 6);
    }

    for (int group = 0; group < 8; group++) {
      int byteOffset = 16 + (group >>> 1) * 32;
      int shift = (group & 1) * 4;
      for (int index = 0; index < 32; index++) {
        block[byteOffset + index] |=
            (byte) ((quantFactory.applyAsInt(group * 32 + index) & 0x0F) << shift);
      }
    }
    return block;
  }

  private static MemorySegment copy(Arena arena, byte[] bytes) {
    MemorySegment segment = arena.allocate(bytes.length);
    MemorySegment.copy(bytes, 0, segment, ValueLayout.JAVA_BYTE, 0, bytes.length);
    return segment;
  }

  private static byte[] concat(byte[] a, byte[] b) {
    byte[] out = new byte[a.length + b.length];
    System.arraycopy(a, 0, out, 0, a.length);
    System.arraycopy(b, 0, out, a.length, b.length);
    return out;
  }

  private static float[] ones(int length) {
    float[] out = new float[length];
    java.util.Arrays.fill(out, 1.0f);
    return out;
  }

  private static float[] patternedQuery(int length) {
    float[] out = new float[length];
    for (int i = 0; i < length; i++) {
      out[i] = (i % 11) * 0.25f - 1.25f;
    }
    return out;
  }

  private static float[] patternedQueries(int batchSize, int cols) {
    float[] out = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        out[batch * cols + col] = (float) Math.sin((batch + 1.0) * (col + 0.25)) * (batch + 0.5f);
      }
    }
    return out;
  }

  private static int[] q4Corrections(int activationEntries) {
    return new int[activationEntries / 4];
  }

  @FunctionalInterface
  private interface PackedByteFactory {
    int packed(int loIndex, int hiIndex);
  }
}
