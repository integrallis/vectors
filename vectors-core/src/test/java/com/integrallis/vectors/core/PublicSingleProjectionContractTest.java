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

/** Direct public single-query entry points preserve layout, output ranges and reusable norms. */
class PublicSingleProjectionContractTest {
  private static final ValueLayout.OfShort HALF =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @ParameterizedTest
  @ValueSource(ints = {16, 32, 64})
  void singlePackedLeftAndRightFollowIndependentDenseLayout(int group) {
    int columns = group * 2, rows = 5;
    float[] input = inputs(1, columns, group);
    byte[] weights = weights(rows, columns);
    MemorySegment w = MemorySegment.ofArray(weights),
        sc = MemorySegment.ofArray(scales(rows, columns / group));
    float[] expected = denseLeft(input, 1, rows, columns, group, weights, sc),
        out = new float[rows];
    VectorUtil.packedInt4GroupMatVec(input, w, sc, rows, columns, group, out);
    assertThat(out).containsExactly(expected);
    float[] shortInput = inputs(1, rows, rows),
        right = new float[columns],
        dense = new float[columns];
    for (int c = 0; c < columns; c++)
      for (int r = 0; r < rows; r++)
        dense[c] +=
            shortInput[r]
                * decode(weights, r * columns + c)
                * half(sc, r * (columns / group) + c / group);
    VectorUtil.packedInt4GroupRightMatVec(shortInput, w, sc, rows, columns, group, right);
    assertThat(right).containsExactly(dense);
    float[] before = out.clone();
    for (int badGroup : new int[] {0, -1, 3}) {
      assertThatIllegalArgumentException()
          .isThrownBy(
              () -> VectorUtil.packedInt4GroupMatVec(input, w, sc, rows, columns, badGroup, out));
      assertThat(out).containsExactly(before);
    }
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVec(
                    new float[columns - 1], w, sc, rows, columns, group, out));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVec(
                    input, w.asSlice(0, w.byteSize() - 1), sc, rows, columns, group, out));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVec(
                    input, w, sc.asSlice(0, sc.byteSize() - 1), rows, columns, group, out));
    assertThat(out).containsExactly(before);
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 3, 4, 7, 16, 37, 257})
  void preparedCosineAndOffsetDistanceMatchIndependentAlgebra(int n) {
    float[] q = new float[n], r = new float[n], qp = new float[n + 7], rp = new float[n + 9];
    Arrays.fill(qp, 321f);
    Arrays.fill(rp, -765f);
    double dot = 0, qn = 0, rn = 0, distance = 0;
    for (int i = 0; i < n; i++) {
      q[i] = (i % 7 + 1) * .25f;
      r[i] = (i % 11 - 4) * .5f;
      qp[i + 3] = q[i];
      rp[i + 5] = r[i];
      dot += q[i] * r[i];
      qn += q[i] * q[i];
      rn += r[i] * r[i];
      double d = q[i] - r[i];
      distance += d * d;
    }
    assertThat(VectorUtil.squareDistance(qp, 3, rp, 5, n)).isEqualTo((float) distance);
    float expected = (float) (dot / Math.sqrt(qn * rn));
    if (n == 0) {
      assertThat(VectorUtil.cosineWithQueryNorm(q, r, 0)).isNaN();
    } else {
      float prepared = VectorUtil.cosineQueryNorm(q);
      assertThat(VectorUtil.cosineWithQueryNorm(q, r, prepared)).isCloseTo(expected, within(1e-6f));
      assertThat(
              VectorUtil.cosineWithQueryNorm(
                  MemorySegment.ofArray(q), MemorySegment.ofArray(r), n, prepared))
          .isCloseTo(expected, within(1e-6f));
      q[0] += 1f;
      assertThat(VectorUtil.cosineWithQueryNorm(q, r, VectorUtil.cosineQueryNorm(q)))
          .isCloseTo(VectorUtil.cosine(q, r), within(1e-6f));
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
}
