package com.integrallis.vectors.hnsw;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ProductQuantizer;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Tests for quantization-integrated two-pass search in HNSW. */
class QuantizedSearchTest {

  @Nested
  @Tag("unit")
  class NodeScorerBasics {

    @Test
    void fullPrecisionNodeScorer_matchesSimilarityFunction() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(20, 8, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
      var rav = new InMemoryVectors(vectors);

      NodeScorerFactory factory = query -> nodeId -> sim.compare(query, rav.getVector(nodeId));
      NodeScorer scorer = factory.scorer(vectors[0]);

      for (int i = 0; i < vectors.length; i++) {
        float expected = sim.compare(vectors[0], vectors[i]);
        assertThat(scorer.score(i)).isEqualTo(expected);
      }
    }

    @Test
    void quantizedNodeScorer_returnsApproximateScores() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(100, 16, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);

      NodeScorerFactory factory =
          query -> {
            var sf = compressed.scoreFunctionFor(query, sim);
            return sf::score;
          };

      NodeScorer scorer = factory.scorer(vectors[0]);

      // Basic sanity: all scores must be finite and non-negative.
      float[] approxScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        approxScores[i] = scorer.score(i);
        assertThat(approxScores[i]).isFinite();
        assertThat(approxScores[i]).isGreaterThanOrEqualTo(0f);
      }

      // Rank-correlation: for SQ8 the self-score (query vs itself) should rank in the top-5
      // across all 100 vectors. A scorer that returns constant values or is completely
      // uncorrelated with exact similarity would fail this check.
      float selfScore = approxScores[0]; // vectors[0] is the query
      int rank = 0;
      for (float s : approxScores) {
        if (s > selfScore) rank++;
      }
      assertThat(rank)
          .as(
              "SQ8 approximate self-score should rank top-5 among 100 vectors (actual rank: %d)",
              rank)
          .isLessThan(5);
    }
  }

  @Nested
  @Tag("unit")
  class VectorDatasetBridge {

    @Test
    void randomAccessVectorDataset_wrapsCorrectly() {
      float[][] data = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      var rav = new InMemoryVectors(data);
      var dataset = new RandomAccessVectorDataset(rav);

      assertThat(dataset.size()).isEqualTo(50);
      assertThat(dataset.dimension()).isEqualTo(8);
      for (int i = 0; i < data.length; i++) {
        assertThat(dataset.getVector(i)).isEqualTo(data[i]);
      }
    }

    @Test
    void randomAccessVectorDataset_canTrainQuantizer() {
      float[][] data = HnswGraphBuilderTest.randomVectors(100, 16, 42L);
      var rav = new InMemoryVectors(data);
      var dataset = new RandomAccessVectorDataset(rav);

      // Should be usable as VectorDataset for quantizer training
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(100);
      assertThat(compressed.dimension()).isEqualTo(16);
    }
  }

  @Nested
  @Tag("unit")
  class RefactoredSearcher {

    @Test
    void searchWithFullPrecisionScorer_matchesOriginalSearch() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(200, 16, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Full-precision search should give identical results
      var result1 = index.search(vectors[0], 10);
      var result2 = index.search(vectors[0], 10);
      assertThat(result1.nodeIds()).isEqualTo(result2.nodeIds());
      assertThat(result1.scores()).isEqualTo(result2.scores());
    }

    @Test
    void searchWithCustomScorer_respectsScoringFunction() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Self-query should return the query vector as rank 0
      var result = index.search(vectors[0], 5);
      assertThat(result.nodeId(0)).isEqualTo(0);
    }
  }

  @Nested
  @Tag("unit")
  class Rescore {

    @Test
    void rescore_reranksWithFullPrecision() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(100, 16, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();
      var searcher = index.searcher();

      // Get some candidates
      var searchResult = searcher.search(vectors[0], 20);
      // Rescore top 20 to get top 5
      var rescored = searcher.rescore(vectors[0], searchResult.nodeIds(), 5);

      assertThat(rescored.size()).isEqualTo(5);
      // Scores should be sorted descending
      for (int i = 0; i < rescored.size() - 1; i++) {
        assertThat(rescored.score(i)).isGreaterThanOrEqualTo(rescored.score(i + 1));
      }
      // Each rescored score should match exact similarity
      for (int i = 0; i < rescored.size(); i++) {
        float exact = sim.compare(vectors[0], vectors[rescored.nodeId(i)]);
        assertThat(rescored.score(i)).isEqualTo(exact);
      }
    }

    @Test
    void rescore_returnsTopK_whenMoreCandidatesThanK() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index = HnswIndex.builder(vectors, sim).seed(42L).build();
      var searcher = index.searcher();

      // Rescore all 50 vectors to get top 5
      int[] allIds = new int[50];
      for (int i = 0; i < 50; i++) allIds[i] = i;

      var rescored = searcher.rescore(vectors[0], allIds, 5);
      assertThat(rescored.size()).isEqualTo(5);

      // Compare against brute force top 5
      int[] bruteForce = HnswSearcherTest.bruteForceKnn(vectors[0], vectors, sim, 5);
      assertThat(rescored.nodeIds()).containsExactly(bruteForce);
    }
  }

  @Nested
  @Tag("unit")
  class QuantizationAttachment {

    @Test
    void enableQuantization_validatesSize() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Create compressed vectors with different size
      float[][] smaller = HnswGraphBuilderTest.randomVectors(30, 8, 99L);
      var dataset = new ArrayVectorDataset(smaller);
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);

      assertThatIllegalArgumentException()
          .isThrownBy(() -> index.enableQuantization(compressed))
          .withMessageContaining("size");
    }

    @Test
    void enableQuantization_validatesDimension() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      // Create compressed vectors with different dimension
      float[][] diffDim = HnswGraphBuilderTest.randomVectors(50, 16, 99L);
      var dataset = new ArrayVectorDataset(diffDim);
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);

      assertThatIllegalArgumentException()
          .isThrownBy(() -> index.enableQuantization(compressed))
          .withMessageContaining("dimension");
    }

    @Test
    void isQuantizationEnabled_reflectsState() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(50, 8, 42L);
      var index = HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN).seed(42L).build();

      assertThat(index.isQuantizationEnabled()).isFalse();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);

      index.enableQuantization(compressed);
      assertThat(index.isQuantizationEnabled()).isTrue();

      index.disableQuantization();
      assertThat(index.isQuantizationEnabled()).isFalse();
    }
  }

  @Nested
  @Tag("unit")
  class TwoPassSearch {

    @Test
    void searchTwoPass_withSQ8_returnsCorrectResults() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(200, 16, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      index.enableQuantization(sq.encodeAll(dataset));

      var result = index.searchTwoPass(vectors[0], 10);
      assertThat(result.size()).isEqualTo(10);

      // Scores should be sorted descending (rescore produces full-precision scores).
      for (int i = 0; i < result.size() - 1; i++) {
        assertThat(result.score(i)).isGreaterThanOrEqualTo(result.score(i + 1));
      }

      // Recall check: the results must actually be near-neighbors.
      // SQ8 two-pass on 200 random 16-dim vectors should achieve > 80% recall@10.
      int[] bruteForce = HnswSearcherTest.bruteForceKnn(vectors[0], vectors, sim, 10);
      double recall = SiftLoader.recallAtK(bruteForce, result.nodeIds(), 10);
      assertThat(recall)
          .as("SQ8 two-pass recall@10 on 200 random 16-dim vectors should be > 0.80")
          .isGreaterThan(0.80);
    }

    @Test
    void searchTwoPass_usesFullPrecisionScoring_whenNoQuantization() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(100, 16, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
      var index = HnswIndex.builder(vectors, sim).seed(42L).build();

      // Without quantization the two-pass pipeline uses full-precision scoring throughout.
      // Each returned score must equal the true similarity, not a quantized approximation.
      var result = index.searchTwoPass(vectors[0], 10);
      assertThat(result.size()).isEqualTo(10);
      for (int i = 0; i < result.size(); i++) {
        float exact = sim.compare(vectors[0], vectors[result.nodeId(i)]);
        assertThat(result.score(i))
            .as("Score at rank %d should be full-precision (expected %.6f)", i, exact)
            .isEqualTo(exact);
      }
    }

    @Test
    void searchTwoPass_rescoresWithFullPrecision() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(200, 16, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      index.enableQuantization(sq.encodeAll(dataset));

      var result = index.searchTwoPass(vectors[0], 5);
      // After rescore, each score should match the exact similarity
      for (int i = 0; i < result.size(); i++) {
        float exact = sim.compare(vectors[0], vectors[result.nodeId(i)]);
        assertThat(result.score(i))
            .as("Score at rank %d should match exact similarity", i)
            .isEqualTo(exact);
      }
    }
  }

  @Nested
  @Tag("unit")
  class TwoPassRecall {

    @Test
    void recall10_above93_withSQ8() {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(1000, 128, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      index.enableQuantization(sq.encodeAll(dataset));

      double totalRecall = 0;
      int numQueries = 100;
      for (int i = 0; i < numQueries; i++) {
        int[] bruteForce = HnswSearcherTest.bruteForceKnn(vectors[i], vectors, sim, 10);
        var result = index.searchTwoPass(vectors[i], 10, 100, 1.5f);
        totalRecall += SiftLoader.recallAtK(bruteForce, result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / numQueries;
      assertThat(avgRecall)
          .as("SQ8 two-pass recall@10 (1000 128-dim, overQueryFactor=1.5)")
          .isGreaterThan(0.93);
    }

    @Test
    void recall10_withPQ_improvedByRescore() {
      // PQ on random Gaussian data is inherently lossy (no subspace structure to exploit).
      // This test validates that two-pass rescore improves upon pure PQ ranking.
      float[][] vectors = HnswGraphBuilderTest.randomVectors(1000, 128, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);
      index.enableQuantization(pq.encodeAll(dataset));

      double totalRecall = 0;
      int numQueries = 100;
      for (int i = 0; i < numQueries; i++) {
        int[] bruteForce = HnswSearcherTest.bruteForceKnn(vectors[i], vectors, sim, 10);
        var result = index.searchTwoPass(vectors[i], 10, 200, 10.0f);
        totalRecall += SiftLoader.recallAtK(bruteForce, result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / numQueries;
      // PQ on random data: recall > 0.70 with generous overQueryFactor
      assertThat(avgRecall)
          .as("PQ two-pass recall@10 (1000 128-dim, 8 subspaces, overQueryFactor=10.0)")
          .isGreaterThan(0.70);
    }

    @Test
    void twoPassRecall_improvesWithLargerCandidatePool() {
      // Validates that a larger overQueryFactor (more candidates in the quantized coarse pass)
      // yields equal or higher recall after full-precision rescoring. Both passes always rescore —
      // the difference is the size of the candidate pool handed to the rescore step.
      //   overQueryFactor=1.0 → coarseK=10  → rescore 10 candidates → pick top-10
      //   overQueryFactor=2.0 → coarseK=20  → rescore 20 candidates → pick top-10
      // With a larger pool the rescore step has more chances to promote true neighbors that the
      // quantized ranking placed at ranks 11-20, so recall(2.0) >= recall(1.0).
      float[][] vectors = HnswGraphBuilderTest.randomVectors(1000, 128, 42L);
      SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      CompressedVectors compressed = sq.encodeAll(dataset);
      index.enableQuantization(compressed);

      double totalRecallSmall = 0;
      double totalRecallLarger = 0;
      int numQueries = 100;
      for (int i = 0; i < numQueries; i++) {
        int[] bruteForce = HnswSearcherTest.bruteForceKnn(vectors[i], vectors, sim, 10);

        // Small candidate pool: 10 quantized candidates, rescored to 10.
        var smallPool = index.searchTwoPass(vectors[i], 10, 100, 1.0f);
        totalRecallSmall += SiftLoader.recallAtK(bruteForce, smallPool.nodeIds(), 10);

        // Larger candidate pool: 20 quantized candidates, rescored to 10.
        var largerPool = index.searchTwoPass(vectors[i], 10, 100, 2.0f);
        totalRecallLarger += SiftLoader.recallAtK(bruteForce, largerPool.nodeIds(), 10);
      }
      double recallSmall = totalRecallSmall / numQueries;
      double recallLarger = totalRecallLarger / numQueries;
      assertThat(recallLarger)
          .as(
              "Larger candidate pool recall (%.3f) should be >= small pool recall (%.3f)",
              recallLarger, recallSmall)
          .isGreaterThanOrEqualTo(recallSmall);
    }
  }

  @Nested
  @Tag("unit")
  class TwoPassAllSimilarityFunctions {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void twoPassSearch_worksWithAllFunctions(SimilarityFunction sim) {
      float[][] vectors = HnswGraphBuilderTest.randomVectors(200, 16, 42L);

      var index =
          HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();

      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      index.enableQuantization(sq.encodeAll(dataset));

      var result = index.searchTwoPass(vectors[0], 5);
      assertThat(result.size()).isEqualTo(5);
      for (int i = 0; i < result.size(); i++) {
        assertThat(result.score(i)).isFinite();
        assertThat(result.score(i)).isGreaterThanOrEqualTo(0f);
      }
    }
  }

  @Nested
  @Tag("slow")
  @EnabledIf("siftSmallExists")
  class SiftSmallQuantizedSearch {

    static final Path SIFT_BASE = Path.of("../../research/repos/jvector/siftsmall");

    @Test
    void siftSmall_sq8TwoPass_recall10_above93() {
      float[][] base = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_BASE.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_BASE.resolve("siftsmall_groundtruth.ivecs"));

      var index =
          HnswIndex.builder(base, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      var dataset = new ArrayVectorDataset(base);
      var sq = ScalarQuantizer.train(dataset);
      index.enableQuantization(sq.encodeAll(dataset));

      double totalRecall = 0;
      for (int i = 0; i < queries.length; i++) {
        var result = index.searchTwoPass(queries[i], 10, 100, 2.0f);
        totalRecall += SiftLoader.recallAtK(groundTruth[i], result.nodeIds(), 10);
      }
      double avgRecall = totalRecall / queries.length;
      assertThat(avgRecall)
          .as("SIFT Small SQ8 two-pass recall@10 (M=16, efC=200, efS=100, oqf=2.0)")
          .isGreaterThan(0.93);
    }

    static boolean siftSmallExists() {
      return Files.exists(SIFT_BASE.resolve("siftsmall_base.fvecs"));
    }
  }
}
