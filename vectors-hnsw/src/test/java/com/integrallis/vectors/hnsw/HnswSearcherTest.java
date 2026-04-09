package com.integrallis.vectors.hnsw;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Tests for HNSW search: basic search, recall validation, and SIFT Small benchmark. */
class HnswSearcherTest {

  @Nested
  @Tag("unit")
  class BasicSearch {

    @Test
    void searchSingleVector_returnsSelf() {
      float[][] vectors = {{1f, 2f, 3f}};
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      var result = index.search(vectors[0], 1);
      assertThat(result.size()).isEqualTo(1);
      assertThat(result.nodeId(0)).isEqualTo(0);
    }

    @Test
    void searchTwoVectors_returnsNearest() {
      float[][] vectors = {
        {0f, 0f, 0f},
        {1f, 0f, 0f},
        {10f, 0f, 0f}
      };
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Query closest to vector 1
      float[] query = {0.9f, 0f, 0f};
      var result = index.search(query, 1);
      assertThat(result.nodeId(0)).isEqualTo(1);
    }

    @Test
    void searchReturnsRequestedK() {
      float[][] vectors = randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      var result = index.search(vectors[0], 10);
      assertThat(result.size()).isEqualTo(10);
    }

    @Test
    void searchResultsSortedByScoreDescending() {
      float[][] vectors = randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      var result = index.search(vectors[0], 10);
      for (int i = 0; i < result.size() - 1; i++) {
        assertThat(result.score(i))
            .as("Score at rank %d should be >= score at rank %d", i, i + 1)
            .isGreaterThanOrEqualTo(result.score(i + 1));
      }
    }

    @Test
    void efSearchLessThanK_throws() {
      float[][] vectors = randomVectors(10, 4, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      assertThatIllegalArgumentException().isThrownBy(() -> index.search(vectors[0], 20, 10));
    }
  }

  @Nested
  @Tag("unit")
  class SmallDatasetSearch {

    @Test
    void recall10_above80_euclidean() {
      float[][] vectors = randomVectors(100, 16, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double avgRecall = computeAverageRecall(vectors, index, SimilarityFunction.EUCLIDEAN, 10, 50);
      assertThat(avgRecall).as("Average recall@10 for euclidean").isGreaterThan(0.80);
    }

    @Test
    void recall10_above80_cosine() {
      float[][] vectors = randomVectors(100, 16, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.COSINE)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double avgRecall = computeAverageRecall(vectors, index, SimilarityFunction.COSINE, 10, 50);
      assertThat(avgRecall).as("Average recall@10 for cosine").isGreaterThan(0.80);
    }

    @Test
    void recallIncreasesWithEfSearch() {
      float[][] vectors = randomVectors(100, 16, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double recallLowEf =
          computeAverageRecall(vectors, index, SimilarityFunction.EUCLIDEAN, 10, 10, 20);
      double recallHighEf =
          computeAverageRecall(vectors, index, SimilarityFunction.EUCLIDEAN, 10, 10, 100);
      assertThat(recallHighEf).isGreaterThanOrEqualTo(recallLowEf);
    }

    @Test
    void selfQuery_returnsQueryAsTopResult() {
      float[][] vectors = randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Every vector should be its own nearest neighbor
      for (int i = 0; i < vectors.length; i++) {
        var result = index.search(vectors[i], 1);
        assertThat(result.nodeId(0)).as("Self-query for vector %d", i).isEqualTo(i);
      }
    }
  }

  @Nested
  @Tag("unit")
  class MediumDatasetSearch {

    @Test
    void recall10_above90_euclidean() {
      float[][] vectors = randomVectors(1000, 128, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double avgRecall =
          computeAverageRecall(vectors, index, SimilarityFunction.EUCLIDEAN, 10, 100);
      assertThat(avgRecall).as("Average recall@10 for 1000 128-dim vectors").isGreaterThan(0.90);
    }

    @Test
    void searchAllQueries_averageRecall_above90() {
      float[][] vectors = randomVectors(1000, 128, 42L);
      float[][] queries = randomVectors(50, 128, 99L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double totalRecall = 0;
      for (float[] query : queries) {
        int[] bruteForce = bruteForceKnn(query, vectors, SimilarityFunction.EUCLIDEAN, 10);
        var result = index.search(query, 10, 100);
        totalRecall += SiftLoader.recallAtK(bruteForce, result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / queries.length;
      assertThat(avgRecall).as("Average recall@10 on external queries").isGreaterThan(0.90);
    }
  }

  @Nested
  @Tag("unit")
  class AllSimilarityFunctions {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void searchWithAllFunctions_returnsFiniteScores(SimilarityFunction sim) {
      float[][] vectors = randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, sim).seed(42L).build();

      var result = index.search(vectors[0], 5);
      for (int i = 0; i < result.size(); i++) {
        assertThat(result.score(i)).isFinite();
        assertThat(result.score(i)).isGreaterThanOrEqualTo(0f);
      }
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void recall_above85_forAllFunctions(SimilarityFunction sim) {
      float[][] vectors = randomVectors(200, 16, 42L);
      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      double avgRecall = computeAverageRecall(vectors, index, sim, 10, 30);
      assertThat(avgRecall).as("recall@10 for " + sim).isGreaterThan(0.85);
    }
  }

  @Nested
  @Tag("unit")
  class HnswIndexApiTests {

    @Test
    void builderDefaults_M16_ef200() {
      float[][] vectors = randomVectors(10, 4, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).build();
      assertThat(index.size()).isEqualTo(10);
      assertThat(index.dimension()).isEqualTo(4);
      assertThat(index.similarityFunction()).isEqualTo(SimilarityFunction.EUCLIDEAN);
    }

    @Test
    void builderCustomParameters() {
      float[][] vectors = randomVectors(20, 4, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.COSINE)
              .maxConnections(32)
              .efConstruction(100)
              .seed(99L)
              .build();
      assertThat(index.size()).isEqualTo(20);
    }

    @Test
    void buildAndSearch_endToEnd() {
      float[][] vectors = randomVectors(100, 16, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      var result = index.search(vectors[0], 5);
      assertThat(result.size()).isEqualTo(5);
      assertThat(result.nodeId(0)).isEqualTo(0); // Self is nearest
    }

    @Test
    void deterministicWithSameSeed() {
      float[][] vectors = randomVectors(100, 16, 42L);

      var index1 = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(99L).build();
      var index2 = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(99L).build();

      var result1 = index1.search(vectors[0], 10);
      var result2 = index2.search(vectors[0], 10);

      assertThat(result1.nodeIds()).isEqualTo(result2.nodeIds());
      assertThat(result1.scores()).isEqualTo(result2.scores());
    }
  }

  @Nested
  @Tag("slow")
  @EnabledIf("siftSmallExists")
  class SiftSmallDataset {

    static final Path SIFT_BASE = Path.of("../../research/repos/jvector/siftsmall");

    @Test
    void siftSmall_euclidean_recall10_above95() {
      float[][] base = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_BASE.resolve("siftsmall_groundtruth.ivecs"));

      var index =
          HnswIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double totalRecall = 0;
      for (int i = 0; i < queries.length; i++) {
        var result = index.search(queries[i], 10, 100);
        totalRecall += SiftLoader.recallAtK(groundTruth[i], result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / queries.length;
      assertThat(avgRecall).as("SIFT Small recall@10 (M=16, efC=200, efS=100)").isGreaterThan(0.95);
    }

    @Test
    void siftSmall_euclidean_recall10_above98_highEf() {
      float[][] base = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_BASE.resolve("siftsmall_groundtruth.ivecs"));

      var index =
          HnswIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double totalRecall = 0;
      for (int i = 0; i < queries.length; i++) {
        var result = index.search(queries[i], 10, 200);
        totalRecall += SiftLoader.recallAtK(groundTruth[i], result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / queries.length;
      assertThat(avgRecall).as("SIFT Small recall@10 (efS=200)").isGreaterThan(0.98);
    }

    @Test
    void siftSmall_recallIncreasesWithEfSearch() {
      float[][] base = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_BASE.resolve("siftsmall_groundtruth.ivecs"));

      var index =
          HnswIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double recallEf50 = averageSiftRecall(index, queries, groundTruth, 10, 50);
      double recallEf100 = averageSiftRecall(index, queries, groundTruth, 10, 100);
      double recallEf200 = averageSiftRecall(index, queries, groundTruth, 10, 200);

      assertThat(recallEf100).isGreaterThanOrEqualTo(recallEf50);
      assertThat(recallEf200).isGreaterThanOrEqualTo(recallEf100);
    }

    @Test
    void siftSmall_graphStatistics() {
      float[][] base = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_base.fvecs"));

      var index =
          HnswIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      assertThat(index.size()).isEqualTo(10_000);
      assertThat(index.dimension()).isEqualTo(128);
    }

    private double averageSiftRecall(
        HnswIndex index, float[][] queries, int[][] groundTruth, int k, int efSearch) {
      double totalRecall = 0;
      for (int i = 0; i < queries.length; i++) {
        var result = index.search(queries[i], k, efSearch);
        totalRecall += SiftLoader.recallAtK(groundTruth[i], result.nodeIds(), k);
      }
      return totalRecall / queries.length;
    }

    static boolean siftSmallExists() {
      return Files.exists(SIFT_BASE.resolve("siftsmall_base.fvecs"));
    }
  }

  // --- Helpers ---

  /** Delegates to the canonical implementation in {@link HnswGraphBuilderTest}. */
  static float[][] randomVectors(int count, int dimension, long seed) {
    return HnswGraphBuilderTest.randomVectors(count, dimension, seed);
  }

  /** Computes average recall using self-queries with default efSearch=100. */
  static double computeAverageRecall(
      float[][] vectors, HnswIndex index, SimilarityFunction sim, int k, int numQueries) {
    return computeAverageRecall(vectors, index, sim, k, numQueries, 100);
  }

  /** Computes average recall using self-queries. */
  static double computeAverageRecall(
      float[][] vectors,
      HnswIndex index,
      SimilarityFunction sim,
      int k,
      int numQueries,
      int efSearch) {
    double totalRecall = 0;
    for (int i = 0; i < numQueries && i < vectors.length; i++) {
      int[] bruteForce = bruteForceKnn(vectors[i], vectors, sim, k);
      var result = index.search(vectors[i], k, efSearch);
      totalRecall += SiftLoader.recallAtK(bruteForce, result.nodeIds(), k);
    }
    return totalRecall / Math.min(numQueries, vectors.length);
  }

  /** Brute-force k-nearest neighbors by scoring all vectors. */
  static int[] bruteForceKnn(float[] query, float[][] vectors, SimilarityFunction sim, int k) {
    var topK = new NodeQueue(k + 1, true); // min-heap
    for (int i = 0; i < vectors.length; i++) {
      float score = sim.compare(query, vectors[i]);
      topK.insertWithOverflow(i, score, k);
    }

    int resultSize = topK.size();
    int[] result = new int[resultSize];
    for (int i = resultSize - 1; i >= 0; i--) {
      result[i] = NodeQueue.nodeId(topK.poll());
    }
    return result;
  }
}
