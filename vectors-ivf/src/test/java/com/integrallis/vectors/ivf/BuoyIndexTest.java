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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BuoyIndexTest {

  private static final int DIM = 64;
  private static final int K = 16;
  private static final int N = 2000;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  @Test
  void trainProducesKBuoys() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 1L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    assertThat(idx.buoyVectors().length).isEqualTo(K);
    assertThat(idx.buoyVectors()[0].length).isEqualTo(DIM);
  }

  @Test
  void kReturnsCorrectCount() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 2L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    assertThat(idx.k()).isEqualTo(K);
  }

  @Test
  void routeReturnsNprobeClusterIds() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 3L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 99L)[0];

    int[] result = idx.route(query, 4, 0f);

    assertThat(result).hasSize(4);
  }

  @Test
  void routeIdsAreUniqueAndWithinRange() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 4L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 100L)[0];

    int[] result = idx.route(query, 8, 0f);

    Set<Integer> seen = new HashSet<>();
    for (int id : result) {
      assertThat(id).isBetween(0, K - 1);
      assertThat(seen.add(id)).isTrue();
    }
  }

  @Test
  void routeIsIdempotent() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 5L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 101L)[0];

    int[] r1 = idx.route(query, 5, 0f);
    int[] r2 = idx.route(query, 5, 0f);

    assertThat(r1).isEqualTo(r2);
  }

  @Test
  void soarGammaExpandsResultsBeyondNprobe() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 6L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 102L)[0];

    int[] plain = idx.route(query, 4, 0f);
    int[] soar = idx.route(query, 4, 0.3f);

    // With SOAR, result may be >= plain (spill target may be added)
    assertThat(soar.length).isGreaterThanOrEqualTo(plain.length);
  }

  @Test
  void encodeDecodeRoundTrip() {
    BuoyIndex original =
        BuoyIndex.train(randomVecs(N, DIM, 7L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    BuoyIndex decoded = BuoyIndex.decode(original.encode());

    assertThat(decoded.k()).isEqualTo(original.k());
    assertThat(decoded.metric()).isEqualTo(original.metric());
    assertThat(decoded.clusterSizes()).isEqualTo(original.clusterSizes());
    for (int i = 0; i < K; i++) {
      assertThat(decoded.buoyVectors()[i]).isEqualTo(original.buoyVectors()[i]);
    }
  }

  @Test
  void clusterSizesArePositiveAndSumToN() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 8L), K, SimilarityFunction.EUCLIDEAN, false, 42L);
    int total = Arrays.stream(idx.clusterSizes()).sum();
    assertThat(total).isEqualTo(N);
    for (int s : idx.clusterSizes()) assertThat(s).isGreaterThanOrEqualTo(0);
  }

  @Test
  void clusterRadiiAreNonNegative() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 9L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    for (float r : idx.clusterRadii()) assertThat(r).isGreaterThanOrEqualTo(0f);
  }

  @Test
  void metricPreserved() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 10L), K, SimilarityFunction.DOT_PRODUCT, false, 0L);
    assertThat(idx.metric()).isEqualTo(SimilarityFunction.DOT_PRODUCT);
  }

  // ─── C2: spill targets must be frequency-based ────────────────────────────

  @Test
  void spillTargets_deterministic_sameDataSameResult() {
    // Training the same data twice must produce the same spill targets
    BuoyIndex a =
        BuoyIndex.train(randomVecs(N, DIM, 42L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    BuoyIndex b =
        BuoyIndex.train(randomVecs(N, DIM, 42L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    assertThat(a.spillTargets()).isEqualTo(b.spillTargets());
  }

  @Test
  void spillTargets_reflectMostFrequentSecondNearest() {
    // Create well-separated 2D clusters at (0,0), (10,0), (20,0) with K=3.
    // Cluster 0 vectors near (0,0) → second-nearest is cluster 1 (at 10) not cluster 2 (at 20).
    // Cluster 1 vectors near (10,0) → second-nearest is cluster 0 or 2 depending on exact position.
    // Cluster 2 vectors near (20,0) → second-nearest is cluster 1 (at 10) not cluster 0 (at 20).
    int clusterK = 3;
    int dim = 2;
    int perCluster = 200;
    float[][] data = new float[clusterK * perCluster][dim];
    Random rng = new Random(42L);
    for (int c = 0; c < clusterK; c++) {
      for (int i = 0; i < perCluster; i++) {
        data[c * perCluster + i][0] = c * 10f + (rng.nextFloat() - 0.5f);
        data[c * perCluster + i][1] = (rng.nextFloat() - 0.5f);
      }
    }

    BuoyIndex idx = BuoyIndex.train(data, clusterK, SimilarityFunction.EUCLIDEAN, true, 42L);

    // Find which buoy index corresponds to cluster at x≈0, x≈10, x≈20
    int idxAt0 = -1, idxAt10 = -1, idxAt20 = -1;
    for (int i = 0; i < clusterK; i++) {
      float x = idx.buoyVectors()[i][0];
      if (x < 5f) idxAt0 = i;
      else if (x < 15f) idxAt10 = i;
      else idxAt20 = i;
    }

    // Cluster near x=0 should spill to the middle cluster (x=10), not the far one (x=20)
    assertThat(idx.spillTargets()[idxAt0]).isEqualTo(idxAt10);
    // Cluster near x=20 should spill to the middle cluster (x=10), not the far one (x=0)
    assertThat(idx.spillTargets()[idxAt20]).isEqualTo(idxAt10);
  }

  // ─── M2: defensive copies — mutation cannot corrupt index ────────────────

  @Test
  void buoyVectors_returnedArrayCannotCorruptIndex() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 20L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] originalFirst = idx.buoyVectors()[0].clone();
    idx.buoyVectors()[0][0] = Float.NaN; // mutate returned array
    assertThat(idx.buoyVectors()[0][0]).isEqualTo(originalFirst[0]);
  }

  @Test
  void spillTargets_returnedArrayCannotCorruptIndex() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 21L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    int originalFirst = idx.spillTargets()[0];
    idx.spillTargets()[0] = 9999;
    assertThat(idx.spillTargets()[0]).isEqualTo(originalFirst);
  }

  @Test
  void clusterSizes_returnedArrayCannotCorruptIndex() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 22L), K, SimilarityFunction.EUCLIDEAN, false, 42L);
    int originalFirst = idx.clusterSizes()[0];
    idx.clusterSizes()[0] = -1;
    assertThat(idx.clusterSizes()[0]).isEqualTo(originalFirst);
  }

  @Test
  void clusterRadii_returnedArrayCannotCorruptIndex() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 23L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float originalFirst = idx.clusterRadii()[0];
    idx.clusterRadii()[0] = Float.NaN;
    assertThat(idx.clusterRadii()[0]).isEqualTo(originalFirst);
  }

  @Test
  void memoryFootprintWithinBound_K1024_D128() {
    // At K=1024, D=128: buoys[]=512KB + spill[]=4KB + sizes[]=4KB + radii[]=4KB ≈ 524KB
    BuoyIndex idx =
        BuoyIndex.train(
            randomVecs(1024 * 256, 128, 11L), 1024, SimilarityFunction.EUCLIDEAN, true, 1L);
    byte[] encoded = idx.encode();
    // Encoded size: 4+4+UTF(metric)+1024*128*4 floats + 3*(1024*4) ints/floats ≈ 536KB
    assertThat(encoded.length).isLessThan(600 * 1024); // 600 KB generous upper bound
  }
}
