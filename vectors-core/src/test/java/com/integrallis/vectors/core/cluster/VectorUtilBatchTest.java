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
package com.integrallis.vectors.core.cluster;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VectorUtilBatchTest {

  private static final int DIM = 128;
  private static final int K = 32;

  private float[][] randomMatrix(int rows, int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[][] m = new float[rows][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private float[] randomVector(int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[] v = new float[dim];
    for (int d = 0; d < dim; d++) v[d] = rng.nextFloat() * 2f - 1f;
    return v;
  }

  private float[] flatten(float[][] matrix, int rows, int cols) {
    float[] flat = new float[rows * cols];
    for (int row = 0; row < rows; row++) {
      System.arraycopy(matrix[row], 0, flat, row * cols, cols);
    }
    return flat;
  }

  @Test
  void batchDotProduct_matchesScalarResults() {
    float[] query = randomVector(DIM, 1L);
    float[][] matrix = randomMatrix(K, DIM, 2L);
    float[] out = new float[K];

    VectorUtil.batchDotProduct(query, matrix, out);

    for (int i = 0; i < K; i++) {
      float expected = VectorUtil.dotProduct(query, matrix[i]);
      assertThat(out[i]).isCloseTo(expected, within(1e-4f));
    }
  }

  @Test
  void batchSquaredL2_matchesScalarResults() {
    float[] query = randomVector(DIM, 3L);
    float[][] matrix = randomMatrix(K, DIM, 4L);
    float[] out = new float[K];

    VectorUtil.batchSquaredL2(query, matrix, out);

    for (int i = 0; i < K; i++) {
      float expected = VectorUtil.squareDistance(query, matrix[i]);
      assertThat(out[i]).isCloseTo(expected, within(1e-4f));
    }
  }

  @Test
  void batchDotProduct_selfDotProduct_equalsSquaredNorm() {
    float[] query = randomVector(DIM, 5L);
    float[][] matrix = new float[][] {query};
    float[] out = new float[1];

    VectorUtil.batchDotProduct(query, matrix, out);

    float expectedNorm = VectorUtil.dotProduct(query, query);
    assertThat(out[0]).isCloseTo(expectedNorm, within(1e-4f));
  }

  @Test
  void batchSquaredL2_sameVector_isZero() {
    float[] query = randomVector(DIM, 6L);
    float[][] matrix = new float[][] {query};
    float[] out = new float[1];

    VectorUtil.batchSquaredL2(query, matrix, out);

    assertThat(out[0]).isCloseTo(0f, within(1e-6f));
  }

  @Test
  void batchDotProduct_emptyMatrix_noOp() {
    float[] query = randomVector(DIM, 7L);
    float[][] matrix = new float[0][];
    float[] out = new float[0];

    VectorUtil.batchDotProduct(query, matrix, out); // must not throw

    assertThat(out).isEmpty();
  }

  @Test
  void rowMajorBatchDotProduct_matchesMatrixRows() {
    int rows = 17;
    int cols = 129;
    float[] query = randomVector(cols, 701L);
    float[][] matrix = randomMatrix(rows, cols, 702L);
    float[] rowMajor = flatten(matrix, rows, cols);
    float[] expected = new float[rows];
    float[] actual = new float[rows];

    VectorUtil.batchDotProduct(query, matrix, expected);
    VectorUtil.batchDotProduct(query, rowMajor, rows, cols, actual);

    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], within(1e-4f));
    }
  }

  @Test
  void rowMajorBatchDotProduct_emptyMatrix_noOp() {
    float[] query = randomVector(8, 703L);
    float[] rowMajor = new float[0];
    float[] out = new float[0];

    VectorUtil.batchDotProduct(query, rowMajor, 0, 8, out);

    assertThat(out).isEmpty();
  }

  @Test
  void batchDotProductRejectsInvalidArguments() {
    float[] query = randomVector(4, 8L);
    float[][] matrix = new float[][] {randomVector(4, 9L)};

    assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, matrix, new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, matrix, new float[1], 2))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                VectorUtil.batchDotProduct(
                    query, new float[][] {randomVector(3, 10L)}, new float[1]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, new float[3], 1, 4, new float[1]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, new float[4], 1, 3, new float[1]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, new float[4], 1, 4, new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void batchSquaredL2RejectsInvalidArguments() {
    float[] query = randomVector(4, 11L);
    float[][] matrix = new float[][] {randomVector(4, 12L)};

    assertThatThrownBy(() -> VectorUtil.batchSquaredL2(query, matrix, new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorUtil.batchSquaredL2(query, matrix, new float[1], -1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                VectorUtil.batchSquaredL2(
                    query, new float[][] {randomVector(3, 13L)}, new float[1]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // --- GEMV edge-case tests (validate 4-row fused kernel corners) ---

  /** K=1, K=2, K=3 — all tail rows (not a full group of 4). */
  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2, 3})
  void matVecDot_tailOnly_matchesScalar(int rows) {
    float[] q = randomVector(DIM, 10L + rows);
    float[][] m = randomMatrix(rows, DIM, 20L + rows);
    float[] out = new float[rows];
    VectorUtil.batchDotProduct(q, m, out);
    for (int i = 0; i < rows; i++) {
      assertThat(out[i]).isCloseTo(VectorUtil.dotProduct(q, m[i]), within(1e-4f));
    }
  }

  /** K=5, K=6, K=7 — partial second group (1-3 tail rows after one full group of 4). */
  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(ints = {5, 6, 7})
  void matVecDot_partialSecondGroup_matchesScalar(int rows) {
    float[] q = randomVector(DIM, 30L + rows);
    float[][] m = randomMatrix(rows, DIM, 40L + rows);
    float[] out = new float[rows];
    VectorUtil.batchDotProduct(q, m, out);
    for (int i = 0; i < rows; i++) {
      assertThat(out[i]).isCloseTo(VectorUtil.dotProduct(q, m[i]), within(1e-4f));
    }
  }

  /** K=1024 (large) — stress test full groups + zero tail. */
  @Test
  void matVecDot_largeK_noTail() {
    int k = 1024;
    float[] q = randomVector(DIM, 99L);
    float[][] m = randomMatrix(k, DIM, 100L);
    float[] out = new float[k];
    VectorUtil.batchDotProduct(q, m, out);
    for (int i = 0; i < k; i++) {
      assertThat(out[i]).isCloseTo(VectorUtil.dotProduct(q, m[i]), within(1e-4f));
    }
  }

  /** Odd dimension (dim not a multiple of 4 or SIMD width) — scalar tail in inner loop. */
  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(ints = {1, 3, 5, 17, 33, 129, 257})
  void matVecDot_oddDimension_matchesScalar(int dim) {
    float[] q = randomVector(dim, 50L);
    float[][] m = randomMatrix(K, dim, 51L);
    float[] out = new float[K];
    VectorUtil.batchDotProduct(q, m, out);
    for (int i = 0; i < K; i++) {
      assertThat(out[i]).isCloseTo(VectorUtil.dotProduct(q, m[i]), within(1e-3f));
    }
  }

  /** matVecSquaredL2 tail + full-group correctness at various row counts. */
  @org.junit.jupiter.params.ParameterizedTest
  @org.junit.jupiter.params.provider.ValueSource(ints = {1, 3, 4, 5, 7, 8, 32})
  void matVecSquaredL2_variousRowCounts_matchesScalar(int rows) {
    float[] q = randomVector(DIM, 60L + rows);
    float[][] m = randomMatrix(rows, DIM, 70L + rows);
    float[] out = new float[rows];
    VectorUtil.batchSquaredL2(q, m, out);
    for (int i = 0; i < rows; i++) {
      assertThat(out[i]).isCloseTo(VectorUtil.squareDistance(q, m[i]), within(1e-3f));
    }
  }
}
