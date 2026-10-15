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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Dense algebra oracles for the public packed-batch wrappers and scalar provider defaults. */
class PublicPackedBatchContractTest {
  private static final ValueLayout.OfShort HALF =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @ParameterizedTest
  @ValueSource(ints = {16, 32, 64})
  void leftAndRightBatchesMatchIndependentDenseProducts(int group) {
    int columns = 2 * group, rows = 7, batches = 3;
    float[] input = inputs(batches, columns, group);
    byte[] packed = weights(rows, columns);
    byte[] scales = scales(rows, columns / group);
    MemorySegment w = MemorySegment.ofArray(packed), s = MemorySegment.ofArray(scales);
    float[] expected = denseLeft(input, batches, rows, columns, group, packed, s);
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      float[] out = guarded(batches * rows);
      provider.packedInt4GroupMatVecBatch(input, batches, w, s, rows, columns, group, out);
      verify(out, expected);
    }
    float[] out = guarded(batches * rows);
    VectorUtil.packedInt4GroupMatVecBatch(input, batches, w, s, rows, columns, group, out);
    verify(out, expected);

    // Transposed operation has scales grouped along the output axis, not the input axis.
    int inputs = 5, outputs = columns;
    float[] rightInput = inputs(batches, inputs, inputs);
    byte[] rightPacked = weights(inputs, outputs);
    MemorySegment rw = MemorySegment.ofArray(rightPacked),
        rs = MemorySegment.ofArray(scales(inputs, outputs / group));
    float[] rightExpected = new float[batches * outputs];
    for (int b = 0; b < batches; b++)
      for (int o = 0; o < outputs; o++) {
        double sum = 0;
        for (int i = 0; i < inputs; i++)
          sum +=
              rightInput[b * inputs + i]
                  * decode(rightPacked, i * outputs + o)
                  * half(rs, i * (outputs / group) + o / group);
        rightExpected[b * outputs + o] = (float) sum;
      }
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      float[] result = guarded(batches * outputs);
      provider.packedInt4GroupRightMatVecBatch(
          rightInput, batches, rw, rs, inputs, outputs, group, result);
      verify(result, rightExpected);
    }
    float[] result = guarded(batches * outputs);
    VectorUtil.packedInt4GroupRightMatVecBatch(
        rightInput, batches, rw, rs, inputs, outputs, group, result);
    verify(result, rightExpected);
  }

  @ParameterizedTest
  @ValueSource(ints = {16, 32, 64})
  void preparedSignedActivationsPreserveLogicalColumnOrder(int group) {
    int columns = 2 * group, rows = 5, batches = 3;
    float[] input = inputs(batches, columns, group);
    byte[] packed = weights(rows, columns);
    MemorySegment w = MemorySegment.ofArray(packed),
        s = MemorySegment.ofArray(scales(rows, columns / group));
    float[] expected = denseLeft(input, batches, rows, columns, group, packed, s);
    byte[] q = new byte[input.length];
    float[] qs = new float[batches * columns / group];
    VectorUtil.quantizeSignedInt8GroupsForPackedInt4(input, batches, columns, group, q, qs);
    // Every group contains 127; all remaining activations are exact integers, so scale is one.
    assertThat(qs).containsOnly(1f);
    for (int b = 0; b < batches; b++)
      for (int c = 0; c < columns; c++) {
        int offset = b * columns + c / group * group;
        int logical = c % group;
        int storage = logical / 2 + (logical % 2) * (group / 2);
        assertThat(q[offset + storage]).isEqualTo((byte) input[b * columns + c]);
      }
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      float[] out = guarded(batches * rows);
      provider.packedInt4GroupMatVecBatchPreparedInt8(
          q, qs, batches, w, s, rows, columns, group, out);
      verify(out, expected);
    }
    float[] out = guarded(batches * rows);
    VectorUtil.packedInt4GroupMatVecBatchPreparedInt8(
        q, qs, batches, w, s, rows, columns, group, out);
    verify(out, expected);
    // Public validation must reject truncated source, scales and output before modifying output.
    float[] before = out.clone();
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVecBatchPreparedInt8(
                    new byte[q.length - 1], qs, batches, w, s, rows, columns, group, out));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVecBatchPreparedInt8(
                    q, new float[qs.length - 1], batches, w, s, rows, columns, group, out));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVecBatch(
                    input, batches, w.asSlice(0, w.byteSize() - 1), s, rows, columns, group, out));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupRightMatVecBatch(
                    input, batches, w, s, rows, columns, 3, out));
    assertThat(out).containsExactly(before);
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 3, 4, 7, 16, 37, 257})
  void reusableQueryNormAndSegmentBatchesMatchDenseAlgebra(int n) {
    float[] q = new float[n], row = new float[n];
    double dot = 0, qn = 0, rn = 0, distance = 0;
    for (int i = 0; i < n; i++) {
      q[i] = (i % 7 + 1) * .25f;
      row[i] = (i % 11 - 4) * .5f;
      dot += q[i] * row[i];
      qn += q[i] * q[i];
      rn += row[i] * row[i];
      double d = q[i] - row[i];
      distance += d * d;
    }
    float cosine = (float) (dot / Math.sqrt(qn * rn));
    MemorySegment qs = MemorySegment.ofArray(q), rs = MemorySegment.ofArray(row);
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      assertThat(provider.cosineWithQueryNorm(q, row, (float) qn)).isCloseTo(cosine, within(1e-6f));
      assertThat(provider.cosineWithQueryNorm(qs, rs, n, (float) qn))
          .isCloseTo(cosine, within(1e-6f));
      float[] out = guarded(2);
      provider.matVecDot(q, new MemorySegment[] {rs, qs}, n, out, 2);
      verify(out, new float[] {(float) dot, (float) qn});
      provider.matVecSquaredL2(q, new MemorySegment[] {rs, qs}, n, out, 2);
      verify(out, new float[] {(float) distance, 0});
    }
  }

  private static float[] inputs(int batches, int columns, int group) {
    float[] a = new float[batches * columns];
    for (int b = 0; b < batches; b++)
      for (int c = 0; c < columns; c++)
        a[b * columns + c] = c % group == 0 ? 127 : (b * 17 + c * 7) % 63 - 31;
    return a;
  }

  private static byte[] weights(int rows, int columns) {
    byte[] a = new byte[rows * columns / 2];
    for (int i = 0; i < rows * columns; i++) {
      int q = (i * 5 + i / columns * 3) % 16 - 8;
      a[i / 2] |= (byte) ((q & 15) << (4 * (i % 2)));
    }
    return a;
  }

  private static byte[] scales(int rows, int groups) {
    byte[] a = new byte[2 * rows * groups];
    MemorySegment s = MemorySegment.ofArray(a);
    for (int i = 0; i < rows * groups; i++)
      s.set(HALF, i * 2L, Float.floatToFloat16((i % 3 + 1) * .5f));
    return a;
  }

  private static float half(MemorySegment s, int i) {
    return Float.float16ToFloat(s.get(HALF, 2L * i));
  }

  private static int decode(byte[] bytes, int index) {
    int nibble = (bytes[index / 2] >>> (4 * (index % 2))) & 15;
    return nibble < 8 ? nibble : nibble - 16;
  }

  private static float[] denseLeft(
      float[] input, int batch, int rows, int columns, int group, byte[] w, MemorySegment scales) {
    float[] result = new float[batch * rows];
    for (int b = 0; b < batch; b++)
      for (int r = 0; r < rows; r++) {
        double sum = 0;
        for (int c = 0; c < columns; c++)
          sum +=
              input[b * columns + c]
                  * decode(w, r * columns + c)
                  * half(scales, r * columns / group + c / group);
        result[b * rows + r] = (float) sum;
      }
    return result;
  }

  private static float[] guarded(int count) {
    float[] a = new float[count + 5];
    Arrays.fill(a, -123456f);
    return a;
  }

  private static void verify(float[] actual, float[] expected) {
    assertThat(Arrays.copyOf(actual, expected.length)).containsExactly(expected);
    assertThat(Arrays.copyOfRange(actual, expected.length, actual.length)).containsOnly(-123456f);
  }
}
