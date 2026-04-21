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

  // -----------------------------------------------------------------------
  // DP1: Fused ADC tests
  // -----------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class FusedAdcSearch {

    private float[][] randomVectors(int n, int dim, long seed) {
      java.util.Random rng = new java.util.Random(seed);
      float[][] vecs = new float[n][dim];
      for (float[] v : vecs) for (int d = 0; d < dim; d++) v[d] = rng.nextFloat() * 2f - 1f;
      return vecs;
    }

    @Test
    void fusedAdcIndex_buildsWithoutError() {
      float[][] vecs = randomVectors(200, 64, 1L);
      // 64-dim, 4 subspaces of 16 each, 256 clusters (standard PQ byte codes)
      var index = HnswFusedAdcIndex.build(vecs, SimilarityFunction.EUCLIDEAN, 16, 100, 4, 256, 42L);
      assertThat(index.size()).isEqualTo(200);
    }

    @Test
    void fusedAdcSearch_returnsSelf() {
      float[][] vecs = randomVectors(100, 32, 2L);
      // 32-dim, 4 subspaces of 8 each, 16 clusters (small for speed)
      var index = HnswFusedAdcIndex.build(vecs, SimilarityFunction.EUCLIDEAN, 16, 100, 4, 16, 42L);
      float[] q = vecs[0].clone();
      var result = index.search(q, 1, 50);
      assertThat(result.size()).isGreaterThanOrEqualTo(1);
    }

    @Test
    void adcTable_scoreOrdering_correlatesWithExactSimilarity() {
      // Direct ADC scoring test — no beam search involved.
      // Constructs a corpus with two clearly-separated clusters (A near +1, B near -1),
      // then verifies that PQ ADC scores rank cluster-A vectors higher than cluster-B
      // vectors when the query is cluster-A's centroid.
      //
      // With well-separated clusters the PQ centroids capture the structure perfectly,
      // so ADC score ordering must match exact ordering without requiring large data.
      int clusterSize = 50;
      int dim = 8; // 2 subspaces of 4 dims each
      java.util.Random rng = new java.util.Random(55L);

      float[][] vecs = new float[clusterSize * 2][dim];
      // Cluster A: centered near +0.8 in every dim (± small noise)
      for (int i = 0; i < clusterSize; i++) {
        for (int d = 0; d < dim; d++) vecs[i][d] = 0.8f + (rng.nextFloat() - 0.5f) * 0.1f;
      }
      // Cluster B: centered near -0.8 in every dim (± small noise)
      for (int i = clusterSize; i < 2 * clusterSize; i++) {
        for (int d = 0; d < dim; d++) vecs[i][d] = -0.8f + (rng.nextFloat() - 0.5f) * 0.1f;
      }

      // Train PQ: 2 subspaces, 4 clusters each — well-separated data ensures clean centroids
      var pq =
          com.integrallis.vectors.quantization.ProductQuantizer.train(
              new com.integrallis.vectors.quantization.ArrayVectorDataset(vecs), 2, 4, false);

      // Query is at the cluster-A centroid
      float[] query = new float[dim];
      java.util.Arrays.fill(query, 0.8f);

      float[][] adcTable = pq.buildADCTable(query, false /* L2 */);

      // Compute ADC raw sum for a cluster-A member (vecs[0]) and a cluster-B member (vecs[50])
      byte[] codeA = pq.encode(vecs[0]);
      byte[] codeB = pq.encode(vecs[clusterSize]);

      float adcSumA = 0f, adcSumB = 0f;
      for (int m = 0; m < 2; m++) {
        adcSumA += adcTable[m][codeA[m] & 0xFF];
        adcSumB += adcTable[m][codeB[m] & 0xFF];
      }

      // For L2: smaller ADC sum = closer = better; adcSumA must be << adcSumB
      assertThat(adcSumA)
          .as(
              "ADC distance to cluster-A member (%.4f) must be < distance to cluster-B member (%.4f)",
              adcSumA, adcSumB)
          .isLessThan(adcSumB);
    }

    /**
     * Production recall gate for Fused ADC two-pass search.
     *
     * <p>Requires n=100,000 vectors for proper PQ training with 256 clusters (39+ vectors per
     * cluster per subspace). Marked {@code @Tag("slow")} to exclude from the default test run.
     */
    @Test
    @org.junit.jupiter.api.Tag("slow")
    void fusedAdcTwoPass_recall_atLeast90pct_vs_bruteForce_large() {
      // n=100k, dim=64, pqSubvectors=4, pqClusters=256 → 100000/4/256 ≈ 97 vectors/cluster ✓
      int n = 100_000;
      int dim = 64;
      int k = 10;
      float[][] vecs = randomVectors(n, dim, 7L);
      float[][] queries = randomVectors(20, dim, 77L);

      var fusedIndex =
          HnswFusedAdcIndex.build(vecs, SimilarityFunction.EUCLIDEAN, 16, 200, 4, 256, 42L);

      int twoPassHits = 0, totalGt = 0;
      for (float[] q : queries) {
        NodeQueue bruteHeap = new NodeQueue(n, true);
        for (int i = 0; i < n; i++) {
          float score = SimilarityFunction.EUCLIDEAN.compare(q, vecs[i]);
          if (bruteHeap.size() < k) bruteHeap.add(i, score);
          else if (score > NodeQueue.score(bruteHeap.peek())) {
            bruteHeap.poll();
            bruteHeap.add(i, score);
          }
        }
        java.util.Set<Integer> gtSet = new java.util.HashSet<>();
        while (!bruteHeap.isEmpty()) gtSet.add(NodeQueue.nodeId(bruteHeap.poll()));

        var twoPass = fusedIndex.searchTwoPass(q, k, 200, 4.0f);
        for (int i = 0; i < twoPass.size(); i++) {
          if (gtSet.contains(twoPass.nodeId(i))) twoPassHits++;
        }
        totalGt += gtSet.size();
      }

      double recall = (double) twoPassHits / totalGt;
      assertThat(recall)
          .as("Fused ADC two-pass recall@10 (large) should be >= 0.90, was %.3f", recall)
          .isGreaterThanOrEqualTo(0.90);
    }

    /**
     * Verifies that scoring a node's entire neighbor list via the batched Fused ADC path ({@code
     * NodeScorer.scoreNeighborBatch}) yields bit-exact results vs scoring each neighbor
     * individually through {@code NodeScorer.score(neighborId)}. This guards the packed-layout byte
     * alignment (neighbor-major {@code [size * M]} bytes).
     */
    @Test
    void fusedAdcBatchScoring_matchesPerNodeScoring() {
      int n = 300;
      int dim = 32;
      float[][] vecs = randomVectors(n, dim, 13L);
      var pq =
          com.integrallis.vectors.quantization.ProductQuantizer.train(
              new com.integrallis.vectors.quantization.ArrayVectorDataset(vecs), 4, 16, true);
      byte[][] allCodes = new byte[n][];
      for (int i = 0; i < n; i++) allCodes[i] = pq.encode(vecs[i]);

      // Build the packed layout for a handful of synthetic "origins".
      int[] origin1Neighbors = {5, 17, 42, 99, 120};
      int[] origin2Neighbors = {0, 1, 2, 3};
      var packed1 = FusedAdcNeighborList.pack(origin1Neighbors, allCodes, pq.numSubspaces());
      var packed2 = FusedAdcNeighborList.pack(origin2Neighbors, allCodes, pq.numSubspaces());

      float[] query = vecs[50].clone();
      float[][] table = pq.buildADCTable(query, false /* L2 */);

      // Batched
      float[] batched1 = new float[origin1Neighbors.length];
      packed1.batchAdcScore(table, batched1);
      float[] batched2 = new float[origin2Neighbors.length];
      packed2.batchAdcScore(table, batched2);

      // Per-node reference via scalar adcScore
      for (int i = 0; i < origin1Neighbors.length; i++) {
        float expected =
            com.integrallis.vectors.core.VectorUtil.assembleAndSum(
                table, allCodes[origin1Neighbors[i]], 0, pq.numSubspaces());
        assertThat(batched1[i]).isCloseTo(expected, within(1e-5f));
      }
      for (int i = 0; i < origin2Neighbors.length; i++) {
        float expected =
            com.integrallis.vectors.core.VectorUtil.assembleAndSum(
                table, allCodes[origin2Neighbors[i]], 0, pq.numSubspaces());
        assertThat(batched2[i]).isCloseTo(expected, within(1e-5f));
      }
    }

    @Test
    void fusedAdcSearch_neighborListSize_matches() {
      float[][] vecs = randomVectors(50, 16, 9L);
      // 16-dim, 2 subspaces × 8 dims, 16 clusters
      var index = HnswFusedAdcIndex.build(vecs, SimilarityFunction.EUCLIDEAN, 8, 50, 2, 16, 42L);
      var result = index.search(vecs[0], 5, 30);
      assertThat(result.size()).isBetween(1, 5);
    }
  }
}
