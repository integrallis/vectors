package com.integrallis.vectors.vamana;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.stream.Stream;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class VamanaSearcherTest {

  @Nested
  @Tag("unit")
  class BasicSearch {

    @Test
    void search_throwsForWrongQueryDimension() {
      float[][] data = {{1f, 2f, 3f}, {4f, 5f, 6f}};
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(2)
              .searchListSize(10)
              .seed(42L)
              .build();

      // Query has 2 dimensions, index expects 3.
      assertThatThrownBy(() -> index.search(new float[] {1f, 2f}, 1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("dimension");
    }

    @Test
    void searchSingleVector_returnsSelf() {
      float[][] data = {{1, 2, 3}};
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(2)
              .searchListSize(10)
              .seed(42L)
              .build();

      SearchResult result = index.search(new float[] {1, 2, 3}, 1);
      assertThat(result.size()).isEqualTo(1);
      assertThat(result.nodeId(0)).isEqualTo(0);
    }

    @Test
    void searchTwoVectors_returnsNearest() {
      // Query is closer to vector 1
      float[][] data = {{0, 0}, {10, 10}};
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(2)
              .searchListSize(10)
              .seed(42L)
              .build();

      SearchResult result = index.search(new float[] {9, 9}, 1);
      assertThat(result.nodeId(0)).isEqualTo(1);
    }

    @Test
    void searchReturnsRequestedK() {
      float[][] data = generateRandomVectors(50, 8, 42L);
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(8)
              .searchListSize(20)
              .seed(42L)
              .build();

      SearchResult result = index.search(data[0], 10);
      assertThat(result.size()).isEqualTo(10);
    }

    @Test
    void searchResultsSortedByScoreDescending() {
      float[][] data = generateRandomVectors(50, 8, 42L);
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(8)
              .searchListSize(20)
              .seed(42L)
              .build();

      SearchResult result = index.search(data[0], 10);
      for (int i = 1; i < result.size(); i++) {
        assertThat(result.score(i - 1))
            .as("Score at rank %d should be >= score at rank %d", i - 1, i)
            .isGreaterThanOrEqualTo(result.score(i));
      }
    }
  }

  @Nested
  @Tag("unit")
  class RecallTests {

    @Test
    void recall10_above80_100vectors_euclidean() {
      float[][] data = generateRandomVectors(100, 16, 42L);
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(32)
              .searchListSize(100)
              .alpha(1.2f)
              .seed(42L)
              .build();

      double avgRecall = computeAverageRecall(data, index, SimilarityFunction.EUCLIDEAN, 10, 100);
      assertThat(avgRecall).as("recall@10 on 100 random 16-dim vectors").isGreaterThan(0.80);
    }

    @Test
    void recall10_above90_1000vectors_euclidean() {
      float[][] data = generateRandomVectors(1000, 128, 42L);
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(64)
              .searchListSize(128)
              .alpha(1.2f)
              .seed(42L)
              .build();

      double avgRecall = computeAverageRecall(data, index, SimilarityFunction.EUCLIDEAN, 10, 128);
      assertThat(avgRecall).as("recall@10 on 1000 random 128-dim vectors").isGreaterThan(0.90);
    }

    @Test
    void selfQuery_returnsQueryAsTopResult() {
      float[][] data = generateRandomVectors(100, 16, 42L);
      var index =
          VamanaIndex.builder(data, SimilarityFunction.EUCLIDEAN)
              .maxDegree(16)
              .searchListSize(50)
              .seed(42L)
              .build();

      // Each vector should be its own nearest neighbor
      for (int i = 0; i < data.length; i++) {
        SearchResult result = index.search(data[i], 1);
        assertThat(result.nodeId(0))
            .as("Self-query for vector %d should return itself", i)
            .isEqualTo(i);
      }
    }
  }

  @Nested
  @Tag("unit")
  class AllSimilarityFunctions {

    static Stream<Arguments> similarityFunctions() {
      return Stream.of(
          Arguments.of(SimilarityFunction.EUCLIDEAN),
          Arguments.of(SimilarityFunction.DOT_PRODUCT),
          Arguments.of(SimilarityFunction.COSINE),
          Arguments.of(SimilarityFunction.MAXIMUM_INNER_PRODUCT));
    }

    @ParameterizedTest
    @MethodSource("similarityFunctions")
    void recall_above85_forAllFunctions(SimilarityFunction sim) {
      float[][] data = generateRandomVectors(200, 32, 42L);

      // For inner-product-based similarities, normalize vectors so scores are well-behaved.
      // EUCLIDEAN is scale-invariant in terms of ranking, so no normalization needed.
      if (sim == SimilarityFunction.DOT_PRODUCT
          || sim == SimilarityFunction.COSINE
          || sim == SimilarityFunction.MAXIMUM_INNER_PRODUCT) {
        for (float[] vec : data) {
          float norm = 0;
          for (float v : vec) {
            norm += v * v;
          }
          norm = (float) Math.sqrt(norm);
          for (int d = 0; d < vec.length; d++) {
            vec[d] /= norm;
          }
        }
      }

      var index =
          VamanaIndex.builder(data, sim)
              .maxDegree(32)
              .searchListSize(100)
              .alpha(1.2f)
              .seed(42L)
              .build();

      double avgRecall = computeAverageRecall(data, index, sim, 10, 100);
      assertThat(avgRecall)
          .as("recall@10 for %s on 200 random 32-dim vectors", sim)
          .isGreaterThan(0.85);
    }
  }

  // --- Helpers ---

  /** Delegates to the canonical implementation in {@link VamanaGraphBuilderTest}. */
  private static float[][] generateRandomVectors(int count, int dim, long seed) {
    return VamanaGraphBuilderTest.generateRandomVectors(count, dim, seed);
  }

  /** Computes average recall@k over all queries using brute-force ground truth. */
  static double computeAverageRecall(
      float[][] data, VamanaIndex index, SimilarityFunction sim, int k, int searchListSize) {
    int n = data.length;
    int numQueries = Math.min(n, 100); // sample up to 100 queries
    double totalRecall = 0;

    for (int q = 0; q < numQueries; q++) {
      float[] query = data[q];

      // Brute-force ground truth
      int[] groundTruth = bruteForceKnn(data, query, k, sim);

      // Approximate search
      SearchResult result = index.search(query, k, searchListSize);
      int[] approx = result.nodeIds();

      totalRecall += SiftLoader.recallAtK(groundTruth, approx, k);
    }
    return totalRecall / numQueries;
  }

  /** Brute-force k-NN: returns top-k node IDs sorted by similarity descending. */
  private static int[] bruteForceKnn(float[][] data, float[] query, int k, SimilarityFunction sim) {
    int n = data.length;
    var resultQueue = new NodeQueue(k + 1, true); // min-heap
    for (int i = 0; i < n; i++) {
      float score = sim.compare(query, data[i]);
      resultQueue.insertWithOverflow(i, score, k);
    }
    int resultSize = resultQueue.size();
    int[] ids = new int[resultSize];
    for (int i = resultSize - 1; i >= 0; i--) {
      ids[i] = NodeQueue.nodeId(resultQueue.poll());
    }
    return ids;
  }
}
