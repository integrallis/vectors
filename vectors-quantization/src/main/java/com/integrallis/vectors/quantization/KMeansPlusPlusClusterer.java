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

  private KMeansPlusPlusClusterer() {}

  /**
   * Clusters a set of sub-vectors into k centroids.
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
}
