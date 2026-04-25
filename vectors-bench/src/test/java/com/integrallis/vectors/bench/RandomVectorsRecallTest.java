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
package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.vamana.VamanaIndex;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Recall regression guards for both HNSW and Vamana on a deterministic random dataset. No external
 * dataset files are required — the corpus and queries are generated from a fixed seed, making this
 * test safe to run in any CI environment.
 *
 * <p>Tests both algorithms on identical data so differences in recall reflect algorithmic
 * behaviour, not dataset variation.
 */
@Tag("slow")
class RandomVectorsRecallTest {

  private static final int N = 10_000;
  private static final int DIMS = 128;
  private static final int N_QUERIES = 200;
  private static final long SEED = 42L;
  private static final int K = 10;

  private static float[][] corpus;
  private static float[][] queries;
  private static int[][] groundTruth;
  private static VamanaIndex vamanaIndex;
  private static HnswIndex hnswIndex;

  @BeforeAll
  static void generateAndBuild() {
    // Generate deterministic corpus and queries.
    Random rng = new Random(SEED);
    corpus = new float[N][DIMS];
    for (float[] v : corpus) {
      for (int d = 0; d < DIMS; d++) v[d] = rng.nextFloat() * 2f - 1f;
    }
    queries = new float[N_QUERIES][DIMS];
    for (float[] q : queries) {
      for (int d = 0; d < DIMS; d++) q[d] = rng.nextFloat() * 2f - 1f;
    }
    groundTruth =
        RecallUtil.bruteForceGroundTruth(queries, corpus, SimilarityFunction.EUCLIDEAN, K);

    // Build both indexes after corpus is available.
    vamanaIndex =
        VamanaIndex.builder(corpus, SimilarityFunction.EUCLIDEAN)
            .maxDegree(64)
            .searchListSize(128)
            .alpha(1.2f)
            .seed(SEED)
            .build();
    hnswIndex =
        HnswIndex.builder(corpus, SimilarityFunction.EUCLIDEAN)
            .maxConnections(16)
            .efConstruction(200)
            .seed(SEED)
            .build();
  }

  // -------------------------------------------------------------------------
  // Vamana
  // -------------------------------------------------------------------------

  @Test
  void vamana_fullPrecision_L128_recall10_above95() {
    double recall = vamanaRecall(K, 128);
    assertThat(recall)
        .as("Vamana full-precision L=128 recall@10 on 10K random 128-dim vectors")
        .isGreaterThanOrEqualTo(0.95);
  }

  @Test
  void vamana_fullPrecision_L200_recall10_above97() {
    double recall = vamanaRecall(K, 200);
    assertThat(recall)
        .as("Vamana full-precision L=200 recall@10 on 10K random 128-dim vectors")
        .isGreaterThanOrEqualTo(0.97);
  }

  // -------------------------------------------------------------------------
  // HNSW
  // -------------------------------------------------------------------------

  @Test
  void hnsw_fullPrecision_ef128_recall10_above90() {
    double recall = hnswRecall(K, 128);
    // Random uniform data is HNSW's hardest case (no cluster structure).
    // M=16 / efConstruction=200 reaches ~0.91 at ef=128 on 10K×128-dim random vectors.
    assertThat(recall)
        .as("HNSW full-precision ef=128 recall@10 on 10K random 128-dim vectors")
        .isGreaterThanOrEqualTo(0.90);
  }

  @Test
  void hnsw_fullPrecision_ef200_recall10_above95() {
    double recall = hnswRecall(K, 200);
    assertThat(recall)
        .as("HNSW full-precision ef=200 recall@10 on 10K random 128-dim vectors")
        .isGreaterThanOrEqualTo(0.95);
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private static double vamanaRecall(int k, int searchListSize) {
    int[][] approx = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      com.integrallis.vectors.vamana.SearchResult r =
          vamanaIndex.search(queries[i], k, searchListSize);
      approx[i] = r.nodeIds();
    }
    return RecallUtil.meanRecallAtK(groundTruth, approx, k);
  }

  private static double hnswRecall(int k, int efSearch) {
    int[][] approx = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      com.integrallis.vectors.hnsw.SearchResult r = hnswIndex.search(queries[i], k, efSearch);
      approx[i] = r.nodeIds();
    }
    return RecallUtil.meanRecallAtK(groundTruth, approx, k);
  }
}
