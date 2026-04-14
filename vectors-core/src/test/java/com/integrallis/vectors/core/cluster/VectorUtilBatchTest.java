package com.integrallis.vectors.core.cluster;

import static org.assertj.core.api.Assertions.assertThat;
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
}
