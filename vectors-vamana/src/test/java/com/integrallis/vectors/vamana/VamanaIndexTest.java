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
package com.integrallis.vectors.vamana;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

class VamanaIndexTest {

  @Nested
  @Tag("unit")
  class BuilderAPI {

    @Test
    void builderDefaults_R64_L128_alpha1_2() {
      float[][] data = generateRandomVectors(20, 4, 42L);
      // Just verify it builds with defaults without error
      VamanaIndex index = VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN).seed(42L).build();
      assertThat(index.size()).isEqualTo(20);
    }

    @Test
    void builderCustomParameters() {
      float[][] data = generateRandomVectors(20, 4, 42L);
      VamanaIndex index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(8)
              .searchListSize(32)
              .alpha(1.5f)
              .seed(42L)
              .build();
      assertThat(index.size()).isEqualTo(20);
      assertThat(index.graph().maxDegree()).isEqualTo(8);
    }

    @Test
    void buildAndSearch_endToEnd() {
      float[][] data = generateRandomVectors(50, 8, 42L);
      VamanaIndex index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(8)
              .searchListSize(20)
              .seed(42L)
              .build();

      SearchResult result = index.search(data[0], 5);
      assertThat(result.size()).isEqualTo(5);
      // Self should be the top result
      assertThat(result.nodeId(0)).isEqualTo(0);
    }

    @Test
    void deterministicWithSameSeed() {
      float[][] data = generateRandomVectors(50, 8, 42L);
      VamanaIndex index1 =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN).seed(99L).build();
      VamanaIndex index2 =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN).seed(99L).build();

      SearchResult r1 = index1.search(data[0], 10);
      SearchResult r2 = index2.search(data[0], 10);
      assertThat(r1.nodeIds()).isEqualTo(r2.nodeIds());
    }
  }

  @Nested
  @Tag("slow")
  @EnabledIf("siftSmallAvailable")
  class SiftSmallDataset {

    private static final Path SIFT_DIR = Path.of("../../research/repos/jvector/siftsmall");

    static boolean siftSmallAvailable() {
      return SIFT_DIR.resolve("siftsmall_base.fvecs").toFile().exists();
    }

    @Test
    void siftSmall_euclidean_recall10_above95() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      VamanaIndex index =
          VamanaIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxDegree(64)
              .searchListSize(128)
              .alpha(1.2f)
              .seed(42L)
              .build();

      double totalRecall = 0;
      int k = 10;
      for (int q = 0; q < queries.length; q++) {
        SearchResult result = index.search(queries[q], k, 128);
        totalRecall += SiftLoader.recallAtK(groundTruth[q], result.nodeIds(), k);
      }
      double avgRecall = totalRecall / queries.length;

      assertThat(avgRecall).as("SIFT Small recall@10 (R=64, L=128, alpha=1.2)").isGreaterThan(0.95);
    }

    @Test
    void siftSmall_recallIncreasesWithSearchListSize() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      VamanaIndex index =
          VamanaIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxDegree(64)
              .searchListSize(128)
              .alpha(1.2f)
              .seed(42L)
              .build();

      int k = 10;
      double recallL64 = computeRecall(queries, groundTruth, index, k, 64);
      double recallL128 = computeRecall(queries, groundTruth, index, k, 128);
      double recallL200 = computeRecall(queries, groundTruth, index, k, 200);

      assertThat(recallL128)
          .as("recall at L=128 should be >= recall at L=64")
          .isGreaterThanOrEqualTo(recallL64);
      assertThat(recallL200)
          .as("recall at L=200 should be >= recall at L=128")
          .isGreaterThanOrEqualTo(recallL128);
      assertThat(recallL200).as("recall at L=200 should be > 0.98").isGreaterThan(0.98);
    }

    private double computeRecall(
        float[][] queries, int[][] groundTruth, VamanaIndex index, int k, int searchListSize) {
      double totalRecall = 0;
      for (int q = 0; q < queries.length; q++) {
        SearchResult result = index.search(queries[q], k, searchListSize);
        totalRecall += SiftLoader.recallAtK(groundTruth[q], result.nodeIds(), k);
      }
      return totalRecall / queries.length;
    }
  }

  // --- Helpers ---

  /** Delegates to the canonical implementation in {@link VamanaGraphBuilderTest}. */
  private static float[][] generateRandomVectors(int count, int dim, long seed) {
    return VamanaGraphBuilderTest.generateRandomVectors(count, dim, seed);
  }
}
