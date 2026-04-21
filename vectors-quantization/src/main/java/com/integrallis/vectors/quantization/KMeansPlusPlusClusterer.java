package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;

/**
 * K-means++ clustering for training Product Quantization codebooks.
 *
 * <p>Implements Arthur &amp; Vassilvitskii (2007) initialization followed by Lloyd's iterations.
 * Training is deterministic given a fixed seed.
 *
 * <p>Package-private — internal implementation detail of this quantization package.
 *
 * <p>The algorithm:
 *
 * <ol>
 *   <li>K-means++ initialization: select first centroid uniformly at random, then each subsequent
 *       centroid with probability proportional to squared distance to nearest existing centroid
 *   <li>Lloyd's iterations: assign points to nearest centroid, recompute centroids as means of
 *       assigned points, repeat until convergence or max iterations
 * </ol>
 */
final class KMeansPlusPlusClusterer {

  /** Default maximum number of Lloyd's iterations. */
  static final int DEFAULT_MAX_ITERATIONS = 10;

  /** Convergence threshold: stop if fewer than this fraction of points change clusters. */
  private static final float CONVERGENCE_THRESHOLD = 0.01f;

  /** Sentinel: no anisotropic refinement — behave as standard unweighted k-means. */
  static final float UNWEIGHTED = -1.0f;

  private KMeansPlusPlusClusterer() {}

  /**
   * Clusters a set of sub-vectors into k centroids using unweighted k-means++.
   *
   * @param data the sub-vectors to cluster, stored as a flat array: {@code data[i * dim + d]} is
   *     dimension d of point i
   * @param numPoints the number of sub-vectors
   * @param dim the dimensionality of each sub-vector
   * @param k the number of clusters (centroids)
   * @param maxIterations maximum Lloyd's iterations
   * @param rng random number generator for initialization
   * @return centroids as a flat array: {@code centroids[c * dim + d]}, size k * dim
   */
  static float[] cluster(
      float[] data, int numPoints, int dim, int k, int maxIterations, Random rng) {
    return cluster(data, numPoints, dim, k, maxIterations, 0, UNWEIGHTED, rng);
  }

  /**
   * Clusters a set of sub-vectors into k centroids with optional anisotropic refinement.
   *
   * <p>When {@code anisotropicThreshold >= 0}, runs {@code maxIterations} unweighted Lloyd's
   * iterations first, then {@code anisotropicIterations} weighted iterations that penalise errors
   * parallel to the data vector direction (ScaNN / AVQ, Guo et al. 2020).
   *
   * @param anisotropicIterations weighted-refinement iterations (0 to disable)
   * @param anisotropicThreshold the T parameter from AVQ \u00a73.4 (typical values 0.1\u20130.3);
   *     pass {@link #UNWEIGHTED} to disable anisotropic refinement entirely
   */
  static float[] cluster(
      float[] data,
      int numPoints,
      int dim,
      int k,
      int maxIterations,
      int anisotropicIterations,
      float anisotropicThreshold,
      Random rng) {
    if (numPoints <= k) {
      // Fewer points than clusters — each point is its own centroid, pad with zeros
      float[] centroids = new float[k * dim];
      System.arraycopy(data, 0, centroids, 0, numPoints * dim);
      return centroids;
    }

    float[] centroids = initializePlusPlus(data, numPoints, dim, k, rng);
    int[] assignments = new int[numPoints];

    for (int iter = 0; iter < maxIterations; iter++) {
      int changed = assignPoints(data, numPoints, dim, centroids, k, assignments);
      updateCentroids(data, numPoints, dim, centroids, k, assignments);

      if ((float) changed / numPoints < CONVERGENCE_THRESHOLD) {
        break;
      }
    }

    if (anisotropicIterations > 0 && anisotropicThreshold >= 0f) {
      refineAnisotropic(
          data,
          numPoints,
          dim,
          centroids,
          k,
          assignments,
          anisotropicIterations,
          anisotropicThreshold);
    }

    return centroids;
  }

  /**
   * K-means++ initialization: selects k centroids from the data with distance-proportional
   * sampling.
   */
  private static float[] initializePlusPlus(
      float[] data, int numPoints, int dim, int k, Random rng) {
    float[] centroids = new float[k * dim];

    // First centroid: random
    int first = rng.nextInt(numPoints);
    System.arraycopy(data, first * dim, centroids, 0, dim);

    // Minimum distance to nearest centroid for each point
    float[] minDist = new float[numPoints];
    for (int i = 0; i < numPoints; i++) {
      minDist[i] = squareDistance(data, i * dim, centroids, 0, dim);
    }

    for (int c = 1; c < k; c++) {
      // Select next centroid with probability proportional to squared distance
      double totalDist = 0;
      for (int i = 0; i < numPoints; i++) {
        totalDist += minDist[i];
      }

      if (totalDist == 0) {
        // All points are identical to existing centroids — pick randomly
        int idx = rng.nextInt(numPoints);
        System.arraycopy(data, idx * dim, centroids, c * dim, dim);
      } else {
        double threshold = rng.nextDouble() * totalDist;
        double cumulative = 0;
        int selected = numPoints - 1;
        for (int i = 0; i < numPoints; i++) {
          cumulative += minDist[i];
          if (cumulative >= threshold) {
            selected = i;
            break;
          }
        }
        System.arraycopy(data, selected * dim, centroids, c * dim, dim);
      }

      // Update minimum distances
      for (int i = 0; i < numPoints; i++) {
        float dist = squareDistance(data, i * dim, centroids, c * dim, dim);
        if (dist < minDist[i]) {
          minDist[i] = dist;
        }
      }
    }

    return centroids;
  }

  /**
   * Assigns each point to the nearest centroid. Returns the number of points that changed
   * assignment.
   */
  private static int assignPoints(
      float[] data, int numPoints, int dim, float[] centroids, int k, int[] assignments) {
    int changed = 0;
    for (int i = 0; i < numPoints; i++) {
      int nearest = 0;
      float nearestDist = squareDistance(data, i * dim, centroids, 0, dim);

      for (int c = 1; c < k; c++) {
        float dist = squareDistance(data, i * dim, centroids, c * dim, dim);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = c;
        }
      }

      if (assignments[i] != nearest) {
        assignments[i] = nearest;
        changed++;
      }
    }
    return changed;
  }

  /**
   * Updates centroids as the mean of assigned points. Empty clusters keep their current centroid.
   */
  private static void updateCentroids(
      float[] data, int numPoints, int dim, float[] centroids, int k, int[] assignments) {
    // Accumulate sums and counts
    float[] sums = new float[k * dim];
    int[] counts = new int[k];

    for (int i = 0; i < numPoints; i++) {
      int c = assignments[i];
      counts[c]++;
      for (int d = 0; d < dim; d++) {
        sums[c * dim + d] += data[i * dim + d];
      }
    }

    // Compute means (skip empty clusters — they keep the existing centroid)
    for (int c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (int d = 0; d < dim; d++) {
          centroids[c * dim + d] = sums[c * dim + d] / counts[c];
        }
      }
    }
  }

  /** Computes squared L2 distance between two sub-vectors in flat arrays. */
  private static float squareDistance(float[] a, int aOff, float[] b, int bOff, int len) {
    return VectorUtil.squareDistance(a, aOff, b, bOff, len);
  }

  // -------------------------------------------------------------------------
  // Anisotropic refinement (ScaNN / AVQ, Guo et al. 2020, Appendix 7.5).
  // Ported from JVector KMeansPlusPlusClusterer (Apache 2.0).
  // -------------------------------------------------------------------------

  /**
   * Runs {@code iterations} of weighted Lloyd's refinement after unweighted seeding. Each iteration
   * reassigns every point to the centroid that minimises the anisotropic loss, then recomputes each
   * centroid by solving a small linear system that weights errors parallel to each point's
   * direction by the parallel-cost multiplier derived from {@code threshold}.
   */
  private static void refineAnisotropic(
      float[] data,
      int numPoints,
      int dim,
      float[] centroids,
      int k,
      int[] assignments,
      int iterations,
      float threshold) {
    if (dim < 2) return; // PCM formula requires dim >= 2
    float pcm = parallelCostMultiplier(threshold, dim);
    float ocm = 1.0f / pcm;
    float[] norms2 = new float[numPoints];
    for (int i = 0; i < numPoints; i++) {
      norms2[i] = VectorUtil.dotProduct(data, i * dim, data, i * dim, dim);
    }
    for (int iter = 0; iter < iterations; iter++) {
      updateCentroidsAnisotropic(data, numPoints, dim, centroids, k, assignments, norms2, ocm);
      int changed = assignAnisotropic(data, numPoints, dim, centroids, k, assignments, norms2, pcm);
      if ((float) changed / numPoints < CONVERGENCE_THRESHOLD) break;
    }
  }

  /**
   * PCM = max(1, (t\u00b2) / ((1 - t\u00b2) / (dim - 1))) — from JVector KMeansPlusPlusClusterer.
   */
  static float parallelCostMultiplier(double threshold, int dimensions) {
    double parallelCost = threshold * threshold;
    double perpendicularCost = (1 - parallelCost) / (dimensions - 1);
    return (float) Math.max(1.0, parallelCost / perpendicularCost);
  }

  /** Anisotropic assignment: each point joins the centroid minimising {@link #weightedDistance}. */
  private static int assignAnisotropic(
      float[] data,
      int numPoints,
      int dim,
      float[] centroids,
      int k,
      int[] assignments,
      float[] norms2,
      float pcm) {
    float[] cNorm2 = new float[k];
    for (int c = 0; c < k; c++) {
      cNorm2[c] = VectorUtil.dotProduct(centroids, c * dim, centroids, c * dim, dim);
    }
    int changed = 0;
    for (int i = 0; i < numPoints; i++) {
      int best = assignments[i];
      float bestDist = Float.MAX_VALUE;
      for (int c = 0; c < k; c++) {
        float d =
            weightedDistance(data, i * dim, centroids, c * dim, dim, pcm, cNorm2[c], norms2[i]);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      if (best != assignments[i]) {
        assignments[i] = best;
        changed++;
      }
    }
    return changed;
  }

  /**
   * d_w(x, c) = pcm * (c\u00b7x - ||x||\u00b2)\u00b2 / ||x||\u00b2 + (||c - x||\u00b2 - that
   * parallel term).
   */
  private static float weightedDistance(
      float[] data,
      int xOff,
      float[] centroids,
      int cOff,
      int dim,
      float pcm,
      float cNorm2,
      float xNorm2) {
    float cDotX = VectorUtil.dotProduct(centroids, cOff, data, xOff, dim);
    float residual2 = cNorm2 - 2f * cDotX + xNorm2;
    // parallel_error = ((c\u00b7x - ||x||\u00b2)\u00b2) / ||x||\u00b2 (unit direction = x / ||x||)
    float parallelNum = cDotX - xNorm2;
    float parallelError = (xNorm2 > 0) ? (parallelNum * parallelNum) / xNorm2 : 0f;
    float perpendicularError = residual2 - parallelError;
    return pcm * parallelError + perpendicularError;
  }

  /**
   * AVQ \u00a77.5 centroid update: c_i = (\u03a3 x_j x_j\u1d40 / ||x_j||\u00b2 * (1 - ocm) / |L| +
   * ocm * I)\u207b\u00b9 * mean(L). For tiny sub-dimensions (common: 16\u201349), direct
   * Gauss\u2013Jordan inversion is cheap enough.
   */
  private static void updateCentroidsAnisotropic(
      float[] data,
      int numPoints,
      int dim,
      float[] centroids,
      int k,
      int[] assignments,
      float[] norms2,
      float ocm) {
    int[] counts = new int[k];
    for (int i = 0; i < numPoints; i++) counts[assignments[i]]++;
    float[][] A = new float[dim][dim];
    float[] mean = new float[dim];
    for (int c = 0; c < k; c++) {
      int size = counts[c];
      if (size == 0) continue; // empty cluster: keep prior centroid
      for (int r = 0; r < dim; r++) {
        java.util.Arrays.fill(A[r], 0f);
      }
      java.util.Arrays.fill(mean, 0f);
      for (int i = 0; i < numPoints; i++) {
        if (assignments[i] != c) continue;
        int base = i * dim;
        for (int r = 0; r < dim; r++) mean[r] += data[base + r];
        float n2 = norms2[i];
        if (n2 <= 0) continue;
        float invN2 = 1f / n2;
        for (int r = 0; r < dim; r++) {
          float xr = data[base + r] * invN2;
          float[] Ar = A[r];
          for (int col = r; col < dim; col++) {
            Ar[col] += xr * data[base + col];
          }
        }
      }
      float scale = (1f - ocm) / size;
      for (int r = 0; r < dim; r++) {
        for (int col = r; col < dim; col++) {
          A[r][col] *= scale;
          if (col != r) A[col][r] = A[r][col]; // symmetrise
        }
        A[r][r] += ocm;
      }
      float invMean = 1f / size;
      for (int r = 0; r < dim; r++) mean[r] *= invMean;
      float[][] inv = OptimizedProductQuantizer.invert(A);
      for (int r = 0; r < dim; r++) {
        float sum = 0f;
        float[] row = inv[r];
        for (int col = 0; col < dim; col++) sum += row[col] * mean[col];
        centroids[c * dim + r] = sum;
      }
    }
  }
}
