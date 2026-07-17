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

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class ScalarVectorUtilSupportTest {

  private final ScalarVectorUtilSupport scalar = new ScalarVectorUtilSupport();

  @Test
  void floatArrayKernelsCoverUnrolledAndTailPaths() {
    float[] a = new float[37];
    float[] b = new float[37];
    for (int i = 0; i < a.length; i++) {
      a[i] = i + 1;
      b[i] = (i % 5) - 2;
    }

    assertThat(scalar.dotProduct(a, b)).isEqualTo(scalar.dotProduct(a, 0, b, 0, a.length));
    assertThat(scalar.dotProduct(a, 3, b, 5, 9)).isEqualTo(referenceDot(a, 3, b, 5, 9));
    assertThat(scalar.squareDistance(a, b)).isEqualTo(scalar.squareDistance(a, 0, b, 0, a.length));
    assertThat(scalar.squareDistance(a, 2, b, 4, 11)).isEqualTo(referenceSquaredL2(a, 2, b, 4, 11));
    assertThat(scalar.cosine(a, a)).isCloseTo(1f, within(1e-6f));
  }

  @Test
  void byteMemoryAndBinaryKernelsReturnReferenceValues() {
    byte[] a = {1, -2, 3, 4};
    byte[] b = {4, 3, -2, 1};
    assertThat(scalar.dotProduct(a, b)).isEqualTo(-4);
    assertThat(scalar.squareDistance(a, b)).isEqualTo(68);
    assertThat(scalar.cosine(a, a)).isCloseTo(1f, within(1e-6f));

    float[] x = {1f, 2f, 3f, 4f, 5f};
    float[] y = {5f, 4f, 3f, 2f, 1f};
    MemorySegment xs = MemorySegment.ofArray(x);
    MemorySegment ys = MemorySegment.ofArray(y);
    assertThat(scalar.dotProduct(xs, ys, x.length)).isEqualTo(35f);
    assertThat(scalar.squareDistance(xs, ys, x.length)).isEqualTo(40f);
    assertThat(scalar.cosine(xs, xs, x.length)).isCloseTo(1f, within(1e-6f));

    assertThat(scalar.hammingDistance(new long[] {0b1010, -1L}, new long[] {0b0011, 0L}))
        .isEqualTo(66);
  }

  @Test
  void arithmeticAndNormalizationHandleEveryContractBranch() {
    float[] value = {1f, 2f, 3f};
    scalar.addInPlace(value, new float[] {3f, 2f, 1f});
    assertThat(value).containsExactly(4f, 4f, 4f);
    scalar.subInPlace(value, new float[] {1f, 2f, 3f});
    assertThat(value).containsExactly(3f, 2f, 1f);
    scalar.scale(value, 2f);
    assertThat(value).containsExactly(6f, 4f, 2f);
    assertThat(scalar.sum(value)).isEqualTo(12f);

    float[] normal = {3f, 4f};
    assertThat(scalar.l2normalize(normal, true)).isSameAs(normal);
    assertThat(normal).containsExactly(0.6f, 0.8f);

    float[] unit = {1f, 0f};
    assertThat(scalar.l2normalize(unit, true)).isSameAs(unit);
    assertThat(unit).containsExactly(1f, 0f);

    float[] zero = {0f, 0f};
    assertThat(scalar.l2normalize(zero, false)).isSameAs(zero);
    assertThatThrownBy(() -> scalar.l2normalize(zero, true))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void fusedAndAdcBatchKernelsCoverFourRowGroupsAndTails() {
    float[] query = {1f, 2f, 3f, 4f, 5f};
    float[][] rows = {
      {1f, 0f, 0f, 0f, 0f},
      {0f, 1f, 0f, 0f, 0f},
      {0f, 0f, 1f, 0f, 0f},
      {0f, 0f, 0f, 1f, 0f},
      {0f, 0f, 0f, 0f, 1f}
    };
    float[] dot = new float[rows.length];
    float[] l2 = new float[rows.length];
    scalar.matVecDot(query, rows, dot, rows.length);
    scalar.matVecSquaredL2(query, rows, l2, rows.length);
    assertThat(dot).containsExactly(1f, 2f, 3f, 4f, 5f);
    for (int i = 0; i < rows.length; i++) {
      assertThat(l2[i]).isEqualTo(referenceSquaredL2(query, 0, rows[i], 0, query.length));
    }

    float[][] table = {{0f, 1f, 2f}, {10f, 11f, 12f}};
    byte[] packed = {1, 2, 0, 1, 2, 0, 1, 1, 0, 2};
    assertThat(scalar.assembleAndSum(table, packed, 0, 2)).isEqualTo(13f);
    float[] adc = new float[5];
    scalar.batchAssembleAndSum(table, packed, 0, adc, 5, 2);
    assertThat(adc).containsExactly(13f, 11f, 12f, 12f, 12f);
  }

  @Test
  void mappedF32GgufKernelReadsLittleEndianRows() {
    float[] query = {1f, 2f, 3f};
    ByteBuffer matrix = ByteBuffer.allocate(6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : new float[] {1f, 0f, -1f, 0.5f, 1f, 2f}) {
      matrix.putFloat(value);
    }
    float[] out = new float[2];

    scalar.ggufF32MatVecDot(query, MemorySegment.ofArray(matrix.array()), 2, 3, out);

    assertThat(out).containsExactly(-2f, 8.5f);
  }

  @Test
  void q4_KBatchedMatmulMatchesIndependentScalarQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    float[] queries = new float[batchSize * cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        queries[batch * cols + col] =
            (float) Math.sin((batch + 1.0) * (col + 0.5)) * (batch + 0.25f);
      }
    }
    byte[] matrix = new byte[rows * (cols / 256) * 144];
    for (int index = 0; index < matrix.length; index++) {
      matrix[index] = (byte) (index * 31 + 7);
    }
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += 144) {
      matrixBuffer.putShort(offset, Float.floatToFloat16(0.01f));
      matrixBuffer.putShort(offset + Short.BYTES, Float.floatToFloat16(0.005f));
    }
    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];
    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      scalar.ggufQ4_KQ8_KMatVecDot(
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

    scalar.ggufQ4_KQ8_KBatchedMatmul(
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
  void q6_KBatchedMatmulMatchesIndependentScalarQueriesExactly() {
    int batchSize = 3;
    int rows = 2;
    int cols = 512;
    int blockBytes = 210;
    float[] queries = new float[batchSize * cols];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = (float) Math.cos((index + 0.5) * 0.03125);
    }
    byte[] matrix = new byte[rows * (cols / 256) * blockBytes];
    for (int index = 0; index < matrix.length; index++) {
      matrix[index] = (byte) (index * 17 + 5);
    }
    ByteBuffer matrixBuffer = ByteBuffer.wrap(matrix).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < matrix.length; offset += blockBytes) {
      matrixBuffer.putShort(offset + 208, Float.floatToFloat16(0.01f));
    }
    MemorySegment weights = MemorySegment.ofArray(matrix);
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];
    float[] query = new float[cols];
    float[] result = new float[rows];
    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(queries, batch * cols, query, 0, cols);
      scalar.ggufQ6_KQ8_KMatVecDot(
          query, weights, rows, cols, result, new byte[cols], new float[cols / 256]);
      System.arraycopy(result, 0, expected, batch * rows, rows);
    }

    scalar.ggufQ6_KQ8_KBatchedMatmul(
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

  private static float referenceDot(float[] a, int ao, float[] b, int bo, int length) {
    float result = 0f;
    for (int i = 0; i < length; i++) result = Math.fma(a[ao + i], b[bo + i], result);
    return result;
  }

  private static float referenceSquaredL2(float[] a, int ao, float[] b, int bo, int length) {
    float result = 0f;
    for (int i = 0; i < length; i++) {
      float delta = a[ao + i] - b[bo + i];
      result += delta * delta;
    }
    return result;
  }

  private static org.assertj.core.data.Offset<Float> within(float value) {
    return org.assertj.core.data.Offset.offset(value);
  }
}
