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
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class Mxfp4MatrixTest {

  private static final float[] E2M1 = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
  };

  @Test
  void decodesTheFreeTokenE2m1TableInLowThenHighNibbleOrder() {
    byte[] blocks = new byte[16];
    for (int packed = 0; packed < 8; packed++) {
      blocks[packed] = (byte) ((2 * packed) | ((2 * packed + 1) << 4));
      blocks[packed + 8] = blocks[packed];
    }
    Mxfp4Matrix matrix =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(blocks), MemorySegment.ofArray(new byte[] {127}), 1, 32);

    for (int column = 0; column < 32; column++) {
      assertThat(matrix.value(0, column)).isEqualTo(E2M1[column % E2M1.length]);
    }
  }

  @Test
  void multipliesPackedRowsWithOneE8m0ScalePerThirtyTwoColumns() {
    byte[] blocks = new byte[32];
    java.util.Arrays.fill(blocks, 0, 16, (byte) 0x21);
    java.util.Arrays.fill(blocks, 16, 32, (byte) 0x77);
    Mxfp4Matrix matrix =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(blocks),
            MemorySegment.ofArray(new byte[] {(byte) 128, (byte) 126}),
            2,
            32);
    float[] output = new float[2];

    matrix.multiply(ones(32), output);

    assertThat(output).containsExactly(48.0f, 96.0f);
  }

  @Test
  void appliesIndependentScalesToConsecutiveBlocks() {
    byte[] blocks = new byte[32];
    java.util.Arrays.fill(blocks, (byte) 0x22);
    Mxfp4Matrix matrix =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(blocks),
            MemorySegment.ofArray(new byte[] {(byte) 127, (byte) 128}),
            1,
            64);
    float[] input = new float[64];
    java.util.Arrays.fill(input, 0, 32, 1.0f);
    java.util.Arrays.fill(input, 32, 64, 0.5f);
    float[] output = new float[1];

    matrix.multiply(input, output);

    assertThat(output[0]).isEqualTo(64.0f);
  }

  @Test
  void slicesRowsWithoutCopying() {
    byte[] blocks = new byte[48];
    java.util.Arrays.fill(blocks, 0, 16, (byte) 0x11);
    java.util.Arrays.fill(blocks, 16, 32, (byte) 0x22);
    java.util.Arrays.fill(blocks, 32, 48, (byte) 0x33);
    Mxfp4Matrix matrix =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(blocks),
            MemorySegment.ofArray(new byte[] {127, 127, 127}),
            3,
            32);

    Mxfp4Matrix slice = matrix.rowSlice(1, 2);
    float[] output = new float[2];
    slice.multiply(ones(32), output);

    assertThat(slice.rows()).isEqualTo(2);
    assertThat(slice.columns()).isEqualTo(32);
    assertThat(output).containsExactly(32.0f, 48.0f);
  }

  @Test
  void q8ActivationKernelStaysCloseToF32ActivationReference() {
    int rows = 32;
    int columns = 64;
    byte[] blocks = new byte[rows * columns / 2];
    byte[] scales = new byte[rows * columns / 32];
    java.util.Random random = new java.util.Random(0x57344138L);
    random.nextBytes(blocks);
    for (int index = 0; index < scales.length; index++) {
      scales[index] = (byte) (125 + random.nextInt(5));
    }
    Mxfp4Matrix matrix =
        Mxfp4Matrix.of(MemorySegment.ofArray(blocks), MemorySegment.ofArray(scales), rows, columns);
    float[] input = new float[columns];
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    float[] expected = new float[rows];
    float[] actual = new float[rows];

    matrix.multiply(input, expected);
    matrix.multiplyQ8(input, actual, GgufQ8_0Batch.allocate(1, columns));

    assertThat(cosine(expected, actual)).isGreaterThan(0.9999);
  }

  @Test
  void reusesOnePreparedQ8ActivationAcrossMatrices() {
    byte[] firstBlocks = new byte[16];
    byte[] secondBlocks = new byte[16];
    java.util.Arrays.fill(firstBlocks, (byte) 0x21);
    java.util.Arrays.fill(secondBlocks, (byte) 0x43);
    Mxfp4Matrix first =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(firstBlocks), MemorySegment.ofArray(new byte[] {127}), 1, 32);
    Mxfp4Matrix second =
        Mxfp4Matrix.of(
            MemorySegment.ofArray(secondBlocks), MemorySegment.ofArray(new byte[] {127}), 1, 32);
    float[] input = new float[32];
    for (int index = 0; index < input.length; index++) {
      input[index] = index - 15.5f;
    }
    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(1, input.length);
    activation.quantize(input, 1);
    float[] firstExpected = new float[1];
    float[] secondExpected = new float[1];
    float[] firstActual = new float[1];
    float[] secondActual = new float[1];

    first.multiplyQ8(input, firstExpected, GgufQ8_0Batch.allocate(1, input.length));
    second.multiplyQ8(input, secondExpected, GgufQ8_0Batch.allocate(1, input.length));
    first.multiplyQ8(activation, firstActual);
    second.multiplyQ8(activation, secondActual);

    assertThat(firstActual).containsExactly(firstExpected);
    assertThat(secondActual).containsExactly(secondExpected);
  }

  @Test
  void nativeStorageMatchesHeapStorageForPreparedQ8Execution() {
    int rows = 7;
    int columns = 96;
    byte[] blocks = new byte[rows * columns / 2];
    byte[] scales = new byte[rows * columns / 32];
    java.util.Random random = new java.util.Random(0x4d584650344e4154L);
    random.nextBytes(blocks);
    random.nextBytes(scales);
    float[] input = new float[columns];
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(1, columns);
    activation.quantize(input, 1);
    float[] expected = new float[rows];
    float[] actual = new float[rows];
    Mxfp4Matrix.of(MemorySegment.ofArray(blocks), MemorySegment.ofArray(scales), rows, columns)
        .multiplyQ8(activation, expected);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment nativeBlocks = arena.allocate(blocks.length);
      MemorySegment nativeScales = arena.allocate(scales.length);
      MemorySegment.copy(blocks, 0, nativeBlocks, ValueLayout.JAVA_BYTE, 0, blocks.length);
      MemorySegment.copy(scales, 0, nativeScales, ValueLayout.JAVA_BYTE, 0, scales.length);
      Mxfp4Matrix.of(nativeBlocks, nativeScales, rows, columns).multiplyQ8(activation, actual);
    }

    assertThat(actual).containsExactly(expected);
  }

  @Test
  void rejectsInvalidGeometryStorageAndInvocation() {
    MemorySegment block = MemorySegment.ofArray(new byte[16]);
    MemorySegment scale = MemorySegment.ofArray(new byte[] {127});
    Mxfp4Matrix matrix = Mxfp4Matrix.of(block, scale, 1, 32);

    assertThatThrownBy(() -> Mxfp4Matrix.of(block, scale, 0, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rows");
    assertThatThrownBy(() -> Mxfp4Matrix.of(block, scale, 1, 31))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("multiple of 32");
    assertThatThrownBy(() -> Mxfp4Matrix.of(block, scale, 2, 32))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blocks");
    assertThatThrownBy(() -> matrix.multiply(new float[31], new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("input");
    assertThatThrownBy(() -> matrix.multiply(new float[32], new float[0]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("output");
    assertThatThrownBy(() -> matrix.rowSlice(1, 1)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  private static float[] ones(int length) {
    float[] values = new float[length];
    java.util.Arrays.fill(values, 1.0f);
    return values;
  }

  private static double cosine(float[] first, float[] second) {
    double dot = 0.0;
    double firstNorm = 0.0;
    double secondNorm = 0.0;
    for (int index = 0; index < first.length; index++) {
      dot += (double) first[index] * second[index];
      firstNorm += (double) first[index] * first[index];
      secondNorm += (double) second[index] * second[index];
    }
    return dot / Math.sqrt(firstNorm * secondNorm);
  }
}
