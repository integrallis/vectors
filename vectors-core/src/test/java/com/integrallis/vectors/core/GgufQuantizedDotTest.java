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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
          query, segment, 2, query.length, out, q8Quants, q8Scales);

      float q8Scale = Float.float16ToFloat(Float.floatToFloat16(1.0f / 127.0f));
      assertThat(out[0]).isCloseTo(189.0f * q8Scale, within(1e-6f));
      assertThat(out[1]).isCloseTo(378.0f * q8Scale, within(1e-6f));
      assertThat(out[0]).isNotEqualTo(1.49f);
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
  void activationQuantizedBatchDotProducts_rejectUndersizedScratch() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q4 = copy(arena, q4Block(1.0f, ones(32), null));
      MemorySegment q8 = copy(arena, q8Block(1.0f));
      MemorySegment q6 = copy(arena, q6KBlock(1.0f, ignored -> 1, ignored -> 1));

      assertThatThrownBy(
              () ->
                  VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
                      ones(32), q4, 1, 32, new float[1], new byte[31], new float[1]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q8Quants.length");
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
  void quantizedDotRejectsNonBlockAlignedDimensions() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q4 = copy(arena, q4Block(1.0f, ones(32), null));
      MemorySegment q8 = copy(arena, q8Block(1.0f));
      MemorySegment q6 = copy(arena, q6KBlock(1.0f, i -> 0, i -> 1));

      assertThatThrownBy(() -> VectorUtil.ggufQ4_0DotProduct(ones(31), q4, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ8_0DotProduct(ones(31), q8, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ6_KDotProduct(ones(255), q6, 0, 255))
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

  @FunctionalInterface
  private interface PackedByteFactory {
    int packed(int loIndex, int hiIndex);
  }
}
