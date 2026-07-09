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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufQuantizedDotTest {

  @Test
  void q4_0DotProduct_matchesDecodedReference() {
    float[] query = new float[32];
    byte[] block = q4Block(0.5f, query, null);

    float expected = 0f;
    for (int i = 0; i < query.length; i++) {
      query[i] = (i % 7) - 3.0f;
      int packed = block[2 + (i >>> 1)] & 0xFF;
      int nibble = (i & 1) == 0 ? packed & 0x0F : (packed >>> 4) & 0x0F;
      expected = Math.fma(query[i], (nibble - 8) * 0.5f, expected);
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = copy(arena, block);

      float actual = VectorUtil.ggufQ4_0DotProduct(query, segment, 0, query.length);

      assertThat(actual).isCloseTo(expected, within(1e-5f));
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
  void quantizedDotRejectsNonBlockAlignedDimensions() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment q4 = copy(arena, q4Block(1.0f, ones(32), null));
      MemorySegment q8 = copy(arena, q8Block(1.0f));

      assertThatThrownBy(() -> VectorUtil.ggufQ4_0DotProduct(ones(31), q4, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
      assertThatThrownBy(() -> VectorUtil.ggufQ8_0DotProduct(ones(31), q8, 0, 31))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("multiple of 32");
    }
  }

  private static byte[] q4Block(float scale, float[] ignored, PackedByteFactory factory) {
    byte[] block = new byte[18];
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, Float.floatToFloat16(scale));
    for (int i = 0; i < 16; i++) {
      if (factory != null) {
        block[2 + i] = (byte) factory.packed(i * 2, i * 2 + 1);
      } else {
        int lo = i & 0x0F;
        int hi = 15 - i;
        block[2 + i] = (byte) (lo | (hi << 4));
      }
    }
    return block;
  }

  private static byte[] q8Block(float scale) {
    byte[] block = new byte[34];
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(0, Float.floatToFloat16(scale));
    for (int i = 0; i < 32; i++) {
      block[2 + i] = (byte) (i - 16);
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

  @FunctionalInterface
  private interface PackedByteFactory {
    int packed(int loIndex, int hiIndex);
  }
}
