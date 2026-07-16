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
import static org.assertj.core.data.Offset.offset;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import jdk.incubator.vector.IntVector;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PanamaGgufQuantizedDotTest {

  @Test
  void q4_0Q8_0IntegerLanesSumFourProductsPerLane() {
    byte[] packed = new byte[16];
    byte[] q8 = new byte[32];
    int[] q4 = new int[32];
    for (int index = 0; index < 16; index++) {
      int low = index % 16;
      int high = 15 - index;
      packed[index] = (byte) (low | (high << 4));
      q4[index] = low - 8;
      q4[index + 16] = high - 8;
    }
    for (int index = 0; index < q8.length; index++) {
      q8[index] = (byte) (index - 16);
    }

    IntVector actual =
        PanamaVectorUtilSupport.q4_0Q8_0IntegerLanes(MemorySegment.ofArray(packed), 0, q8, 0);

    int[] expected = new int[8];
    for (int lane = 0; lane < expected.length; lane++) {
      for (int index = lane * 4; index < lane * 4 + 4; index++) {
        expected[lane] += q4[index] * q8[index];
      }
    }
    assertThat(actual.toArray()).containsExactly(expected);
  }

  @Test
  void q4_KQ8_KIntegerLanesSumFourProductsPerLane() {
    byte[] packed = new byte[32];
    byte[] q8 = new byte[32];
    int[] q4 = new int[32];
    for (int index = 0; index < packed.length; index++) {
      int value = (index * 3) % 16;
      packed[index] = (byte) ((value << 4) | (15 - value));
      q4[index] = value;
      q8[index] = (byte) (index - 16);
    }

    IntVector actual =
        PanamaVectorUtilSupport.q4_KQ8_KIntegerLanes(MemorySegment.ofArray(packed), 0, 4, q8, 0);

    int[] expected = new int[8];
    for (int lane = 0; lane < expected.length; lane++) {
      for (int index = lane * 4; index < lane * 4 + 4; index++) {
        expected[lane] += q4[index] * q8[index];
      }
    }
    assertThat(actual.toArray()).containsExactly(expected);
  }

  @Test
  void q5_KQ8_KIntegerDotDecodesPerElementHighBits() {
    byte[] weights = new byte[64];
    byte[] q8 = new byte[32];
    int highBit = 1 << 3;
    for (int index = 0; index < 32; index++) {
      int low = index * 3 % 16;
      int high = index * 5 % 16;
      weights[32 + index] = (byte) (low | (high << 4));
      if ((index & 1) != 0) {
        weights[index] = (byte) highBit;
      }
      q8[index] = (byte) (index - 16);
    }

    int actual =
        PanamaVectorUtilSupport.q5_KQ8_KIntegerDot(
            MemorySegment.ofArray(weights), 32, 4, 0, highBit, q8, 0);

    int expected = 0;
    for (int index = 0; index < 32; index++) {
      int packed = weights[32 + index] & 0xFF;
      int quant = (packed >>> 4) & 0x0F;
      if ((weights[index] & highBit) != 0) {
        quant += 16;
      }
      expected += quant * q8[index];
    }
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  void q6_KQ8_KIntegerDotDecodesPackedWeightsAndSignedScales() {
    byte[] weights = new byte[80];
    byte[] q8 = new byte[128];
    Random random = new Random(0x516B48L);
    random.nextBytes(weights);
    random.nextBytes(q8);
    int quantOffset = 7;
    int s1 = -11;
    int s2 = 7;
    int s3 = -3;
    int s4 = 13;

    int actual =
        PanamaVectorUtilSupport.q6_KQ8_KIntegerDot(
            MemorySegment.ofArray(weights), 0, 32, 64, q8, quantOffset, s1, s2, s3, s4);

    int expected = 0;
    for (int index = 0; index < 16; index++) {
      int ql1 = weights[index] & 0xFF;
      int ql2 = weights[32 + index] & 0xFF;
      int high = weights[64 + index] & 0xFF;
      int q1 = ((ql1 & 0x0F) | ((high & 0x03) << 4)) - 32;
      int q2 = ((ql2 & 0x0F) | (((high >>> 2) & 0x03) << 4)) - 32;
      int q3 = ((ql1 >>> 4) | (((high >>> 4) & 0x03) << 4)) - 32;
      int q4 = ((ql2 >>> 4) | (((high >>> 6) & 0x03) << 4)) - 32;
      expected += s1 * q1 * q8[quantOffset + index];
      expected += s2 * q2 * q8[quantOffset + index + 32];
      expected += s3 * q3 * q8[quantOffset + index + 64];
      expected += s4 * q4 * q8[quantOffset + index + 96];
    }
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  void q5_0Q8_0IntegerLanesDecodeHighBits() {
    byte[] packed = new byte[16];
    byte[] q8 = new byte[32];
    Random random = new Random(0x515048L);
    random.nextBytes(packed);
    random.nextBytes(q8);
    int highBits = 0xA5C33C5A;

    IntVector actual =
        PanamaVectorUtilSupport.q5_0Q8_0IntegerLanes(
            MemorySegment.ofArray(packed), 0, highBits, q8, 0);

    int[] expected = new int[8];
    for (int index = 0; index < packed.length; index++) {
      int value = packed[index] & 0xFF;
      int low = (value & 0x0F) | (((highBits >>> index) & 1) << 4);
      int high = (value >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
      int lane = index & 7;
      expected[lane] += (low - 16) * q8[index];
      expected[lane] += (high - 16) * q8[index + 16];
    }
    assertThat(actual.toArray()).containsExactly(expected);
  }

  @Test
  void panamaProviderOwnsQ4_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_0Q8_0MatVecDot");
  }

  @Test
  void panamaProviderOwnsQ8_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ8_0Q8_0MatVecDot");
  }

  @Test
  void panamaProviderOwnsQ4_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ5_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ5_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ6_KQ8_KKernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ6_KQ8_KMatVecDot");
  }

  @Test
  void panamaProviderOwnsQ5_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ5_0Q8_0MatVecDot");
  }

  @Test
  void q4_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5148L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 18];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 18) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).hasSameSizeAs(expected);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], offset(1e-4f));
    }
  }

  @Test
  void q8_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5188L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 34];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 34) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q4_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x514B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 144];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 144) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      float minScale = random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(minScale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    short[] expectedSums = new short[cols / 16];
    short[] actualSums = new short[cols / 16];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales,
            expectedSums);
    new PanamaVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales, actualSums);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actualSums).containsExactly(expectedSums);
    assertThat(actual).hasSameSizeAs(expected);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], offset(1e-3f));
    }
  }

  @Test
  void q5_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x515B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 176];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 176) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      float minScale = random.nextFloat() * 0.05f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
      buffer.putShort(offset + Short.BYTES, Float.floatToFloat16(minScale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    short[] expectedSums = new short[cols / 16];
    short[] actualSums = new short[cols / 16];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ5_KQ8_KMatVecDot(
            query,
            weightSegment,
            rows,
            cols,
            expected,
            expectedQuants,
            expectedScales,
            expectedSums);
    new PanamaVectorUtilSupport()
        .ggufQ5_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales, actualSums);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actualSums).containsExactly(expectedSums);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q6_KQ8_KKernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x516B48L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 256) * 210];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 210) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset + 208, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 256];
    float[] actualScales = new float[cols / 256];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).hasSameSizeAs(expected);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], offset(1e-3f));
    }
  }

  @Test
  void q5_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x515048L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 22];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 22) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ5_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ5_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }
}
