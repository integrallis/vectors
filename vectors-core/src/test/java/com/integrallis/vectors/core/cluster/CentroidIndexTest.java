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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CentroidIndexTest {

  private static final int DIM = 64;
  private static final int K = 32;

  private float[][] randomMatrix(int rows, int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[][] m = new float[rows][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  @Test
  void routeReturnsNprobeIds() {
    float[][] centroids = randomMatrix(K, DIM, 1L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 99L)[0];

    int[] result = idx.route(query, 8);

    assertThat(result).hasSize(8);
  }

  @Test
  void routeIdsAreUniqueAndWithinRange() {
    float[][] centroids = randomMatrix(K, DIM, 2L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 100L)[0];

    int[] result = idx.route(query, 10);

    Set<Integer> seen = new HashSet<>();
    for (int id : result) {
      assertThat(id).isBetween(0, K - 1);
      assertThat(seen.add(id)).isTrue();
    }
  }

  @Test
  void routeResultsSortedByAscendingDistance_euclidean() {
    float[][] centroids = randomMatrix(K, DIM, 3L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 101L)[0];

    int[] result = idx.route(query, K);

    for (int r = 0; r < result.length - 1; r++) {
      float d1 = VectorUtil.squareDistance(query, centroids[result[r]]);
      float d2 = VectorUtil.squareDistance(query, centroids[result[r + 1]]);
      assertThat(d1).isLessThanOrEqualTo(d2 + 1e-4f);
    }
  }

  @Test
  void nearestCentroidIsAlwaysFirst() {
    float[][] centroids = randomMatrix(K, DIM, 4L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 102L)[0];

    int nearest = idx.route(query, 1)[0];

    // Verify it's truly nearest
    float nearestDist = VectorUtil.squareDistance(query, centroids[nearest]);
    for (int c = 0; c < K; c++) {
      float d = VectorUtil.squareDistance(query, centroids[c]);
      assertThat(nearestDist).isLessThanOrEqualTo(d + 1e-4f);
    }
  }

  @Test
  void nprobeEqualToKReturnsAllCentroids() {
    float[][] centroids = randomMatrix(K, DIM, 5L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 103L)[0];

    int[] result = idx.route(query, K);

    assertThat(result).hasSize(K);
    // Check all centroid ids appear
    int[] sorted = Arrays.copyOf(result, result.length);
    Arrays.sort(sorted);
    for (int i = 0; i < K; i++) assertThat(sorted[i]).isEqualTo(i);
  }

  @Test
  void soarSpillExpandsResultsBeyondNprobe() {
    // Create centroids with a known near-boundary centroid
    float[][] centroids = new float[4][2];
    centroids[0] = new float[] {1f, 0f}; // nearest to query
    centroids[1] = new float[] {1.05f, 0f}; // just inside boundary (1+gamma)*dist
    centroids[2] = new float[] {-1f, 0f};
    centroids[3] = new float[] {0f, -1f};

    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = new float[] {0f, 0f};

    // spillTargets: cluster 0 spills to cluster 1
    int[] spillTargets = new int[] {1, -1, -1, -1};
    int[] withoutSpill = idx.route(query, 1);
    int[] withSpill = idx.routeWithSpill(query, 1, 0.2f, spillTargets);

    assertThat(withoutSpill).hasSize(1);
    assertThat(withSpill.length).isGreaterThan(1); // spill target added
    assertThat(withSpill).contains(0); // original probe still present
  }

  @Test
  void noSpillWhenGammaIsZero() {
    float[][] centroids = randomMatrix(K, DIM, 6L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 104L)[0];
    int[] spillTargets = new int[K];
    Arrays.fill(spillTargets, 0); // all spill to centroid 0

    int[] result = idx.routeWithSpill(query, 8, 0f, spillTargets);

    assertThat(result).hasSize(8); // no expansion when gamma=0
  }

  @Test
  void dotProductMetric_nearestByHighestDot() {
    float[][] centroids = randomMatrix(K, DIM, 7L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.DOT_PRODUCT);
    float[] query = randomMatrix(1, DIM, 105L)[0];

    int nearest = idx.route(query, 1)[0];

    float nearestDot = VectorUtil.dotProduct(query, centroids[nearest]);
    for (int c = 0; c < K; c++) {
      float dot = VectorUtil.dotProduct(query, centroids[c]);
      assertThat(nearestDot).isGreaterThanOrEqualTo(dot - 1e-4f);
    }
  }

  @Test
  void routeRejectsNonPositiveNprobe() {
    float[][] centroids = randomMatrix(K, DIM, 9L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    float[] query = randomMatrix(1, DIM, 106L)[0];

    assertThatThrownBy(() -> idx.route(query, 0)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> idx.route(query, -1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void cosineMetricRoutesByDirectionNotMagnitude() {
    float[][] centroids = new float[][] {{100f, 0f}, {0f, 1f}};
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.COSINE);

    int nearest = idx.route(new float[] {0f, 2f}, 1)[0];

    assertThat(nearest).isEqualTo(1);
  }

  @Test
  void dotProductSpillUsesScoreSpaceBoundary() {
    float[][] centroids = new float[][] {{10f, 0f}, {9f, 0f}, {1f, 0f}};
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.DOT_PRODUCT);
    int[] spillTargets = new int[] {1, -1, -1};

    int[] result = idx.routeWithSpill(new float[] {1f, 0f}, 1, 0.2f, spillTargets);

    assertThat(result).containsExactly(0, 1);
  }

  @Test
  void constructorRejectsRaggedCentroids() {
    float[][] centroids = new float[][] {{1f, 2f}, {3f}};

    assertThatThrownBy(() -> new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void constructorDefensivelyCopiesCentroids() {
    float[][] centroids = new float[][] {{0f, 0f}, {10f, 0f}};
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    centroids[0][0] = 100f;

    assertThat(idx.route(new float[] {0f, 0f}, 1)[0]).isEqualTo(0);
  }

  @Test
  void centroidCountAndDimension() {
    float[][] centroids = randomMatrix(K, DIM, 8L);
    CentroidIndex idx = new CentroidIndex(centroids, SimilarityFunction.EUCLIDEAN);
    assertThat(idx.centroidCount()).isEqualTo(K);
    assertThat(idx.dimension()).isEqualTo(DIM);
  }
}
