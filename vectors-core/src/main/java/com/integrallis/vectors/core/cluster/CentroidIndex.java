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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/**
 * Centroid routing index: given a query, finds the {@code nprobe} nearest centroids.
 *
 * <p>Supports SOAR boundary expansion: when {@code gamma > 0}, primary probes inside the metric's
 * spill boundary have their spill targets appended to the result. Euclidean routing expands squared
 * L2 distance by {@code (1 + gamma)}. Dot-style routing expands in score space.
 *
 * <p>The index owns a defensive copy of the centroid matrix. For cosine routing, copied centroids
 * are L2-normalized at construction time and queries are normalized before scoring.
 *
 * <p>Distance semantics:
 *
 * <ul>
 *   <li>{@link SimilarityFunction#EUCLIDEAN} — squared L2 (ascending = nearest)
 *   <li>{@link SimilarityFunction#COSINE} — negated cosine (ascending = highest cosine = nearest)
 *   <li>Dot-style metrics — negated dot product (ascending = highest dot = nearest centroid)
 * </ul>
 */
public final class CentroidIndex {

  private final float[][] centroids;
  private final SimilarityFunction metric;

  public CentroidIndex(float[][] centroids, SimilarityFunction metric) {
    this.metric = Objects.requireNonNull(metric, "metric");
    this.centroids = copyAndValidateCentroids(centroids, metric);
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
   * gamma > 0}.
   *
   * @param query the query vector
   * @param nprobe number of primary probes
   * @param gamma boundary expansion factor (0.0 = disabled)
   * @param spillTargets per-centroid spill target (spillTargets[i] = secondary probe for centroid
   *     i); may be null when gamma == 0
   */
  public int[] routeWithSpill(float[] query, int nprobe, float gamma, int[] spillTargets) {
    return routeWithSpillDistances(query, nprobe, gamma, spillTargets).clusterIds();
  }

  /**
   * Route with per-cluster distances preserved. The returned arrays are parallel and sorted by
   * ascending distance for primary probes; spill targets (when {@code gamma > 0}) are appended
   * after, in the order they were added. Distances are in the internal metric scale: squared L2 for
   * EUCLIDEAN, negated cosine for COSINE, and negated dot for dot-style metrics.
   */
  public RouteResult routeWithSpillDistances(
      float[] query, int nprobe, float gamma, int[] spillTargets) {
    int k = centroids.length;
    validateRouteArguments(query, nprobe, gamma, spillTargets, k);
    int probes = Math.min(nprobe, k);
    float[] distances = new float[k];
    computeDistances(query, distances);

    // Partial-sort: primitive top-probes via a bounded max-heap on distance[i].
    int[] topIds = new int[probes];
    float[] topDists = new float[probes];
    int heapSize = 0;
    for (int i = 0; i < k; i++) {
      float d = distances[i];
      if (heapSize < probes) {
        topIds[heapSize] = i;
        topDists[heapSize] = d;
        heapSize++;
        // sift up (max-heap on dist)
        int idx = heapSize - 1;
        while (idx > 0) {
          int parent = (idx - 1) >>> 1;
          if (topDists[idx] > topDists[parent]) {
            float tf = topDists[idx];
            topDists[idx] = topDists[parent];
            topDists[parent] = tf;
            int ti = topIds[idx];
            topIds[idx] = topIds[parent];
            topIds[parent] = ti;
            idx = parent;
          } else break;
        }
      } else if (d < topDists[0]) {
        topIds[0] = i;
        topDists[0] = d;
        // sift down
        int idx = 0;
        int half = heapSize >>> 1;
        while (idx < half) {
          int left = (idx << 1) + 1;
          int right = left + 1;
          int largest = left;
          if (right < heapSize && topDists[right] > topDists[left]) largest = right;
          if (topDists[idx] >= topDists[largest]) break;
          float tf = topDists[idx];
          topDists[idx] = topDists[largest];
          topDists[largest] = tf;
          int ti = topIds[idx];
          topIds[idx] = topIds[largest];
          topIds[largest] = ti;
          idx = largest;
        }
      }
    }
    // Sort (id, dist) pairs ascending by dist — tiny (≤ nprobe) so insertion sort suffices.
    for (int i = 1; i < heapSize; i++) {
      float d = topDists[i];
      int id = topIds[i];
      int j = i - 1;
      while (j >= 0 && topDists[j] > d) {
        topDists[j + 1] = topDists[j];
        topIds[j + 1] = topIds[j];
        j--;
      }
      topDists[j + 1] = d;
      topIds[j + 1] = id;
    }

    boolean[] included = new boolean[k];
    int[] result = new int[Math.min(k, probes * 2 + 1)];
    float[] resultDists = new float[result.length];
    int count = 0;
    for (int r = 0; r < heapSize; r++) {
      result[count] = topIds[r];
      resultDists[count] = topDists[r];
      included[topIds[r]] = true;
      count++;
    }

    if (gamma > 0f && spillTargets != null && count > 0) {
      float nearestDist = topDists[0];
      float boundary = spillBoundary(nearestDist, gamma);
      for (int r = 0; r < heapSize; r++) {
        if (topDists[r] > boundary) break;
        int spill = spillTargets[topIds[r]];
        if (spill >= 0 && spill < k && !included[spill]) {
          if (count >= result.length) {
            result = Arrays.copyOf(result, result.length * 2);
            resultDists = Arrays.copyOf(resultDists, result.length);
          }
          result[count] = spill;
          resultDists[count] = distances[spill];
          included[spill] = true;
          count++;
        }
      }
    }

    return new RouteResult(Arrays.copyOf(result, count), Arrays.copyOf(resultDists, count));
  }

  /**
   * Parallel (clusterId, distance) arrays returned by {@link #routeWithSpillDistances(float[], int,
   * float, int[])}.
   */
  public record RouteResult(int[] clusterIds, float[] distances) {}

  // --- internals ---

  private void computeDistances(float[] query, float[] out) {
    if (metric == SimilarityFunction.EUCLIDEAN) {
      VectorUtil.batchSquaredL2(query, centroids, out);
    } else if (metric == SimilarityFunction.COSINE) {
      // Degrade gracefully on a zero-norm query (degenerate embedding) instead of throwing: a zero
      // query must not crash the whole COSINE search. l2normalize(v, false) leaves a zero vector
      // unchanged → all-zero dots → all centroids equidistant → arbitrary (but valid) routing, and
      // the per-cluster scan returns 0 scores (guarded there). throwOnZero=true would AIOOBE the
      // caller mid-search.
      float[] normalizedQuery = Arrays.copyOf(query, query.length);
      VectorUtil.l2normalize(normalizedQuery, false);
      VectorUtil.batchDotProduct(normalizedQuery, centroids, out);
      for (int i = 0; i < out.length; i++) out[i] = -out[i];
    } else {
      // For DOT_PRODUCT and MIP: higher dot = nearer centroid; negate for ascending sort.
      VectorUtil.batchDotProduct(query, centroids, out);
      for (int i = 0; i < out.length; i++) out[i] = -out[i];
    }
  }

  private float spillBoundary(float nearestDist, float gamma) {
    if (metric == SimilarityFunction.EUCLIDEAN) {
      return nearestDist * (1f + gamma);
    }
    float nearestScore = -nearestDist;
    float thresholdScore = nearestScore - Math.abs(nearestScore) * gamma;
    return -thresholdScore;
  }

  private static float[][] copyAndValidateCentroids(float[][] source, SimilarityFunction metric) {
    Objects.requireNonNull(source, "centroids");
    if (source.length == 0) throw new IllegalArgumentException("empty centroids");
    float[] first = Objects.requireNonNull(source[0], "centroids[0]");
    int dim = first.length;
    if (dim == 0) throw new IllegalArgumentException("centroid dimension must be > 0");

    float[][] copy = new float[source.length][dim];
    for (int i = 0; i < source.length; i++) {
      float[] row = Objects.requireNonNull(source[i], "centroids[" + i + "]");
      if (row.length != dim) {
        throw new IllegalArgumentException(
            "centroid dimensions differ: row 0 has "
                + dim
                + " but row "
                + i
                + " has "
                + row.length);
      }
      copy[i] = Arrays.copyOf(row, dim);
      if (metric == SimilarityFunction.COSINE) {
        VectorUtil.l2normalize(copy[i], true);
      }
    }
    return copy;
  }

  private void validateRouteArguments(
      float[] query, int nprobe, float gamma, int[] spillTargets, int k) {
    Objects.requireNonNull(query, "query");
    if (query.length != dimension()) {
      throw new IllegalArgumentException(
          "query dimension differs from centroids: " + query.length + " != " + dimension());
    }
    if (nprobe <= 0) {
      throw new IllegalArgumentException("nprobe must be > 0: " + nprobe);
    }
    if (!Float.isFinite(gamma) || gamma < 0f) {
      throw new IllegalArgumentException("gamma must be finite and >= 0: " + gamma);
    }
    if (gamma > 0f && spillTargets != null && spillTargets.length < k) {
      throw new IllegalArgumentException(
          "spillTargets.length must be >= centroid count: " + spillTargets.length + " < " + k);
    }
  }
}
