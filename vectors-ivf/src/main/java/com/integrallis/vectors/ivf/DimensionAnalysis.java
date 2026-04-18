package com.integrallis.vectors.ivf;

import java.util.Arrays;

/**
 * Utility for dimension-variance analysis used by the HARMONY partial-distance pruning strategy.
 *
 * <p>For a set of vectors within a single IVF cluster, the dimensions with the <em>highest
 * variance</em> are the most discriminative: a large partial squared-distance computed over only
 * those dimensions is the tightest lower bound for the full squared-distance, maximising the
 * probability that the lower bound exceeds the current k-th neighbour threshold and the candidate
 * can be skipped without a full distance computation.
 *
 * @see IvfIndex
 * @see ClusterPartition#keyDimensions()
 */
public final class DimensionAnalysis {

  private DimensionAnalysis() {}

  /**
   * Returns the indices of the {@code topK} dimensions with the highest per-dimension variance
   * across the given vector subset.
   *
   * <p>The returned array is sorted in <em>descending variance order</em> so the caller can use a
   * prefix to trade recall for speed (fewer key dimensions → less pruning work but also a looser
   * lower bound).
   *
   * @param subVectors vectors belonging to one cluster; may be a subset view (references into the
   *     global array — no copy needed)
   * @param topK number of key dimensions to return; clamped to {@code [1, dimension]}
   * @return dimension indices sorted by descending variance; never null, length = {@code topK}
   * @throws IllegalArgumentException if {@code subVectors} is empty
   */
  public static int[] topVarianceDimensions(float[][] subVectors, int topK) {
    if (subVectors == null || subVectors.length == 0) {
      throw new IllegalArgumentException("subVectors must not be empty");
    }
    int dim = subVectors[0].length;
    topK = Math.max(1, Math.min(topK, dim));
    int n = subVectors.length;

    // Single-pass mean + sum-of-squares (Welford's online algorithm for numerical stability)
    double[] mean = new double[dim];
    double[] m2 = new double[dim]; // sum of squared deviations (Welford M2)

    for (int i = 0; i < n; i++) {
      float[] v = subVectors[i];
      for (int d = 0; d < dim; d++) {
        double delta = v[d] - mean[d];
        mean[d] += delta / (i + 1);
        double delta2 = v[d] - mean[d];
        m2[d] += delta * delta2;
      }
    }

    // Variance = M2 / n (population variance — sufficient for ranking purposes)
    // Build an index array sorted by descending variance
    Integer[] idx = new Integer[dim];
    for (int d = 0; d < dim; d++) idx[d] = d;
    final double[] variance = m2; // M2 / n is monotone with M2 for fixed n
    Arrays.sort(idx, (a, b) -> Double.compare(variance[b], variance[a]));

    int[] result = new int[topK];
    for (int i = 0; i < topK; i++) result[i] = idx[i];
    return result;
  }

  /**
   * Computes the partial squared-Euclidean distance between {@code query} and {@code candidate}
   * using only the given {@code keyDimensions}.
   *
   * <p>This is always ≤ the full squared-Euclidean distance, making it a valid lower bound for
   * EUCLIDEAN pruning.
   *
   * @param query query vector
   * @param candidate candidate vector
   * @param keyDimensions dimension indices to include in the partial computation
   * @return partial squared-Euclidean distance over {@code keyDimensions}
   */
  public static float partialSquaredDistance(
      float[] query, float[] candidate, int[] keyDimensions) {
    float sum = 0f;
    for (int d : keyDimensions) {
      float diff = query[d] - candidate[d];
      sum += diff * diff;
    }
    return sum;
  }
}
