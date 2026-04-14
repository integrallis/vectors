package com.integrallis.vectors.core.cluster;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;

/**
 * Centroid routing index: given a query, finds the {@code nprobe} nearest centroids.
 *
 * <p>Supports SOAR boundary expansion: when {@code gamma > 0}, any centroid whose distance to the
 * query is within {@code (1 + gamma) * nearestDistance} has its spill target appended to the result
 * (up to {@code maxResults} unique ids).
 *
 * <p>Distance semantics:
 *
 * <ul>
 *   <li>{@link SimilarityFunction#EUCLIDEAN} — squared L2 (ascending = nearest)
 *   <li>All others — negated dot product (ascending = highest dot = nearest centroid)
 * </ul>
 */
public final class CentroidIndex {

  private final float[][] centroids;
  private final SimilarityFunction metric;

  public CentroidIndex(float[][] centroids, SimilarityFunction metric) {
    if (centroids.length == 0) throw new IllegalArgumentException("empty centroids");
    this.centroids = centroids;
    this.metric = metric;
  }

  /** Number of centroids. */
  public int centroidCount() {
    return centroids.length;
  }

  /** Dimensionality of the centroid vectors. */
  public int dimension() {
    return centroids[0].length;
  }

  /**
   * Returns the ids of the {@code nprobe} nearest centroids, sorted ascending by distance to {@code
   * query}.
   */
  public int[] route(float[] query, int nprobe) {
    return routeWithSpill(query, nprobe, 0f, null);
  }

  /**
   * Returns centroid ids sorted by ascending distance, expanded by SOAR spill targets when {@code
   * gamma > 0}. At most {@code Math.min(nprobe + spillExpansion, k)} unique ids are returned.
   *
   * @param query the query vector
   * @param nprobe number of primary probes
   * @param gamma boundary expansion factor (0.0 = disabled)
   * @param spillTargets per-centroid spill target (spillTargets[i] = secondary probe for centroid
   *     i); may be null when gamma == 0
   */
  public int[] routeWithSpill(float[] query, int nprobe, float gamma, int[] spillTargets) {
    int k = centroids.length;
    int probes = Math.min(nprobe, k);
    float[] distances = new float[k];
    computeDistances(query, distances);

    // Build index array sorted by ascending distance
    Integer[] order = new Integer[k];
    for (int i = 0; i < k; i++) order[i] = i;
    Arrays.sort(order, (a, b) -> Float.compare(distances[a], distances[b]));

    // Collect primary probes
    boolean[] included = new boolean[k];
    int[] result = new int[Math.min(k, probes * 2 + 1)]; // over-allocate for spill
    int count = 0;
    for (int r = 0; r < probes && r < k; r++) {
      int id = order[r];
      result[count++] = id;
      included[id] = true;
    }

    // SOAR boundary expansion
    if (gamma > 0f && spillTargets != null && count > 0) {
      float nearestDist = distances[order[0]];
      float boundary = nearestDist * (1f + gamma);
      for (int r = 0; r < probes && r < k; r++) {
        int id = order[r];
        if (distances[id] > boundary) break;
        int spill = spillTargets[id];
        if (spill >= 0 && spill < k && !included[spill]) {
          if (count >= result.length) result = Arrays.copyOf(result, result.length * 2);
          result[count++] = spill;
          included[spill] = true;
        }
      }
    }

    return Arrays.copyOf(result, count);
  }

  // --- internals ---

  private void computeDistances(float[] query, float[] out) {
    if (metric == SimilarityFunction.EUCLIDEAN) {
      VectorUtil.batchSquaredL2(query, centroids, out);
    } else {
      // For DOT_PRODUCT, COSINE, MIP: higher dot = nearer centroid → negate for ascending sort
      VectorUtil.batchDotProduct(query, centroids, out);
      for (int i = 0; i < out.length; i++) out[i] = -out[i];
    }
  }
}
