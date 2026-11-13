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

import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EMeansTest {

  private static float[][] randomDataset(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) {
      for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    }
    return m;
  }

  @Test
  void trainProducesKCentroids() {
    float[][] data = randomDataset(500, 32, 1L);
    float[][] centroids = EMeans.train(data, 8, 5, 42L);
    assertThat(centroids.length).isEqualTo(8);
    assertThat(centroids[0].length).isEqualTo(32);
  }

  @Test
  void trainWithNeqK() {
    float[][] data = randomDataset(8, 4, 2L);
    float[][] centroids = EMeans.train(data, 8, 3, 42L);
    assertThat(centroids.length).isEqualTo(8);
  }

  @Test
  void trainDeterministicWithSameSeed() {
    float[][] data = randomDataset(1000, 64, 3L);
    float[][] a = EMeans.train(data, 16, 5, 123L);
    float[][] b = EMeans.train(data, 16, 5, 123L);
    for (int i = 0; i < a.length; i++) {
      assertThat(a[i]).containsExactly(b[i]);
    }
  }

  @Test
  void trainDifferentSeedGivesDifferentResult() {
    float[][] data = randomDataset(1000, 64, 4L);
    float[][] a = EMeans.train(data, 16, 5, 1L);
    float[][] b = EMeans.train(data, 16, 5, 2L);
    boolean allEqual = true;
    outer:
    for (int i = 0; i < a.length; i++) {
      for (int d = 0; d < a[i].length; d++) {
        if (Float.compare(a[i][d], b[i][d]) != 0) {
          allEqual = false;
          break outer;
        }
      }
    }
    assertThat(allEqual).isFalse();
  }

  @Test
  void quantizationErrorLowerThanRandom() {
    // E-means should cluster better than random centroids.
    float[][] data = randomDataset(2000, 32, 5L);
    float[][] emCentroids = EMeans.train(data, 16, 5, 42L);
    float[][] randomCentroids = randomDataset(16, 32, 6L);

    double emError = quantizationError(data, emCentroids);
    double randError = quantizationError(data, randomCentroids);
    assertThat(emError).isLessThan(randError);
  }

  @Test
  void lossDecreasesWithMoreEpochs() {
    float[][] data = randomDataset(2000, 32, 7L);
    float[][] c3 = EMeans.train(data, 16, 3, 42L);
    float[][] c5 = EMeans.train(data, 16, 5, 42L);
    double e3 = quantizationError(data, c3);
    double e5 = quantizationError(data, c5);
    assertThat(e5).isLessThanOrEqualTo(e3);
  }

  private static double quantizationError(float[][] data, float[][] centroids) {
    double total = 0.0;
    float[] scratch = new float[centroids.length];
    for (float[] row : data) {
      int nearest = EMeansTest.nearestCentroid(row, centroids, scratch);
      total += scratch[nearest];
    }
    return total / data.length;
  }

  private static int nearestCentroid(float[] point, float[][] centroids, float[] scratch) {
    int best = -1;
    float bestDist = Float.MAX_VALUE;
    for (int i = 0; i < centroids.length; i++) {
      float[] c = centroids[i];
      float dist = 0.0f;
      for (int d = 0; d < c.length; d++) {
        float diff = point[d] - c[d];
        dist += diff * diff;
      }
      scratch[i] = dist;
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }
}
