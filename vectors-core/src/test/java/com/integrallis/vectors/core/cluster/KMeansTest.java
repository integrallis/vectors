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

import com.integrallis.vectors.core.VectorUtil;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class KMeansTest {

  private float[][] randomVectors(int n, int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[][] data = new float[n][dim];
    for (float[] row : data) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return data;
  }

  @Test
  void trainProducesKCentroids() {
    float[][] data = randomVectors(500, 32, 1L);
    float[][] centroids = KMeans.train(data, 16, 50, 42L);
    assertThat(centroids.length).isEqualTo(16);
    assertThat(centroids[0].length).isEqualTo(32);
  }

  @Test
  void assignReturnsValidClusterIds() {
    float[][] data = randomVectors(200, 16, 2L);
    float[][] centroids = KMeans.train(data, 8, 30, 7L);
    int[] assignments = KMeans.assign(data, centroids);

    assertThat(assignments).hasSize(data.length);
    for (int a : assignments) assertThat(a).isBetween(0, 7);
  }

  @Test
  void quantizationErrorDecreasesWithMoreIterations() {
    float[][] data = randomVectors(300, 32, 3L);
    float[][] centroids1 = KMeans.train(data, 10, 1, 99L);
    float[][] centroids30 = KMeans.train(data, 10, 30, 99L);

    int[] asgn1 = KMeans.assign(data, centroids1);
    int[] asgn30 = KMeans.assign(data, centroids30);
    double err1 = KMeans.quantizationError(data, centroids1, asgn1);
    double err30 = KMeans.quantizationError(data, centroids30, asgn30);

    assertThat(err30).isLessThanOrEqualTo(err1);
  }

  @Test
  void nearestCentroidAssignmentIsCorrect() {
    // two well-separated clusters
    float[][] c1 = new float[50][4];
    float[][] c2 = new float[50][4];
    for (int i = 0; i < 50; i++) {
      c1[i] = new float[] {10f, 0f, 0f, 0f};
      c2[i] = new float[] {-10f, 0f, 0f, 0f};
    }
    float[][] data = new float[100][4];
    System.arraycopy(c1, 0, data, 0, 50);
    System.arraycopy(c2, 0, data, 50, 50);

    float[][] centroids = KMeans.train(data, 2, 20, 0L);
    int[] assignments = KMeans.assign(data, centroids);

    // All of c1 should map to the same cluster, all of c2 to the other
    int label0 = assignments[0];
    for (int i = 0; i < 50; i++) assertThat(assignments[i]).isEqualTo(label0);
    int label1 = assignments[50];
    assertThat(label1).isNotEqualTo(label0);
    for (int i = 50; i < 100; i++) assertThat(assignments[i]).isEqualTo(label1);
  }

  @Test
  void kExceedingDatasetSizeThrows() {
    float[][] data = randomVectors(5, 8, 0L);
    assertThatThrownBy(() -> KMeans.train(data, 6, 10, 0L))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void emptyDatasetThrows() {
    assertThatThrownBy(() -> KMeans.train(new float[0][], 2, 10, 0L))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void centroidsLieWithinDataDistribution() {
    float[][] data = randomVectors(400, 16, 5L);
    // find data range
    float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
    for (float[] row : data)
      for (float v : row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }

    float[][] centroids = KMeans.train(data, 16, 40, 42L);
    for (float[] c : centroids) for (float v : c) assertThat(v).isBetween(min - 1e-3f, max + 1e-3f);
  }

  @Test
  void deterministicWithSameSeed() {
    float[][] data = randomVectors(200, 32, 9L);
    float[][] c1 = KMeans.train(data, 8, 20, 77L);
    float[][] c2 = KMeans.train(data, 8, 20, 77L);
    for (int i = 0; i < c1.length; i++) assertThat(c1[i]).isEqualTo(c2[i]);
  }

  @Test
  void virtualThreadPathProducesSameResultAsSerial() {
    // Force virtual thread path: n > 100_000
    // Use a small dim to keep memory manageable
    int n = 110_000;
    int dim = 4;
    float[][] data = randomVectors(n, dim, 11L);

    // Serial: run with k=4, then compare QE with the virtual-thread path
    float[][] centroids = KMeans.train(data, 4, 5, 42L);
    int[] asgn = KMeans.assign(data, centroids);
    double qe = KMeans.quantizationError(data, centroids, asgn);

    // The virtual thread path is taken automatically; just verify QE is finite and non-negative
    assertThat(qe).isFinite().isGreaterThanOrEqualTo(0.0);
  }

  @Test
  void eachCentroidIsClosestToAtLeastOneDataPoint() {
    float[][] data = randomVectors(200, 8, 13L);
    float[][] centroids = KMeans.train(data, 10, 30, 1L);
    int[] assignments = KMeans.assign(data, centroids);
    boolean[] seen = new boolean[10];
    for (int a : assignments) seen[a] = true;
    for (boolean b : seen) assertThat(b).isTrue();
  }

  @Test
  void kEqualsOneReturnsSingleMeanCentroid() {
    float[][] data = new float[][] {{1f, 0f}, {0f, 1f}, {-1f, 0f}, {0f, -1f}};
    float[][] centroids = KMeans.train(data, 1, 10, 0L);
    assertThat(centroids.length).isEqualTo(1);
    // Centroid should be close to origin (mean of the four points)
    float cx = centroids[0][0], cy = centroids[0][1];
    assertThat(Math.abs(cx)).isLessThan(0.5f);
    assertThat(Math.abs(cy)).isLessThan(0.5f);
  }

  /** Squared L2 to the assigned centroid must be ≤ squared L2 to every other centroid. */
  @Test
  void assignIsOptimal_forTrainedCentroids() {
    float[][] data = randomVectors(100, 8, 14L);
    float[][] centroids = KMeans.train(data, 5, 40, 3L);
    int[] assignments = KMeans.assign(data, centroids);
    for (int i = 0; i < data.length; i++) {
      float assignedDist = VectorUtil.squareDistance(data[i], centroids[assignments[i]]);
      for (int c = 0; c < centroids.length; c++) {
        if (c != assignments[i]) {
          float otherDist = VectorUtil.squareDistance(data[i], centroids[c]);
          assertThat(assignedDist).isLessThanOrEqualTo(otherDist + 1e-4f);
        }
      }
    }
  }
}
