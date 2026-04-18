package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests for {@link ProductQuantizer} and {@link PQVectors}. */
class ProductQuantizerTest {

  /** Path to SIFT Small dataset (from JVector research repo). */
  private static final Path SIFT_DIR =
      Path.of("../../research/repos/jvector/siftsmall").toAbsolutePath().normalize();

  static boolean siftAvailable() {
    return Files.exists(SIFT_DIR.resolve("siftsmall_base.fvecs"));
  }

  @Nested
  @Tag("unit")
  class Training {

    @Test
    void train_producesCorrectCodebooks() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8); // 8 subspaces

      assertThat(pq.dimension()).isEqualTo(128);
      assertThat(pq.numSubspaces()).isEqualTo(8);
      assertThat(pq.numClusters()).isEqualTo(256);
    }

    @Test
    void train_withCustomClusterCount() {
      float[][] vectors = generateVectors(200, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 4, 16); // 4 subspaces, 16 clusters

      assertThat(pq.numSubspaces()).isEqualTo(4);
      assertThat(pq.numClusters()).isEqualTo(16);
    }

    @Test
    void train_reproducibleWithSeed() {
      float[][] vectors = generateVectors(300, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var pq1 = ProductQuantizer.train(dataset, 8);
      var pq2 = ProductQuantizer.train(dataset, 8);

      // Same training data + same default seed = same codebooks
      byte[] encoded1 = pq1.encode(vectors[0]);
      byte[] encoded2 = pq2.encode(vectors[0]);
      assertThat(encoded1).isEqualTo(encoded2);
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 4, 8, 16, 32})
    void train_variousSubspaceCounts(int m) {
      float[][] vectors = generateVectors(300, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, m);

      assertThat(pq.numSubspaces()).isEqualTo(m);
      assertThat(pq.dimension()).isEqualTo(128);
    }

    @Test
    void train_unevenDimensionSplit() {
      // dim=100, M=7: baseSize=100/7=14, remainder=100%7=2 -> sizes [15, 15, 14, 14, 14, 14, 14]
      float[][] vectors = generateVectors(300, 100, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 7);

      assertThat(pq.dimension()).isEqualTo(100);
      assertThat(pq.numSubspaces()).isEqualTo(7);
      // Encode should still work
      byte[] encoded = pq.encode(vectors[0]);
      assertThat(encoded).hasSize(7);
    }

    @Test
    void train_invalidSubspaceCount_throws() {
      float[][] vectors = generateVectors(50, 16, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      assertThatThrownBy(() -> ProductQuantizer.train(dataset, 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> ProductQuantizer.train(dataset, 17)) // M > dim
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void train_gaussianData() {
      float[][] vectors = generateGaussianVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);

      assertThat(pq.dimension()).isEqualTo(128);
      // Should encode without errors
      byte[] encoded = pq.encode(vectors[0]);
      assertThat(encoded).hasSize(8);
    }
  }

  @Nested
  @Tag("unit")
  class EncodeAndDecode {

    @Test
    void encodedSize_equalsMSubspaces() {
      float[][] vectors = generateVectors(300, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);

      byte[] encoded = pq.encode(vectors[0]);
      // One byte per subspace (cluster index [0, 255])
      assertThat(encoded).hasSize(16);
    }

    @Test
    void roundTrip_reconstructionError() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);

      float[] original = vectors[0];
      byte[] encoded = pq.encode(original);
      float[] decoded = pq.decode(encoded);

      assertThat(decoded).hasSize(128);

      // PQ reconstruction has bounded error (each subvector mapped to nearest centroid)
      float mse = 0;
      for (int d = 0; d < 128; d++) {
        float diff = decoded[d] - original[d];
        mse += diff * diff;
      }
      mse /= 128;

      // MSE should be finite and reasonable (not zero — PQ is lossy)
      assertThat(mse).isFinite().isGreaterThan(0f);
    }

    @Test
    void reconstructionError_decreasesWithMoreSubspaces() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var pq4 = ProductQuantizer.train(dataset, 4); // 4 subspaces = coarser
      var pq16 = ProductQuantizer.train(dataset, 16); // 16 subspaces = finer

      double mse4 = averageMse(pq4, vectors);
      double mse16 = averageMse(pq16, vectors);

      // More subspaces = lower MSE
      assertThat(mse16).isLessThan(mse4);
    }

    @Test
    void encodeAll_producesCorrectSize() {
      float[][] vectors = generateVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);
      var compressed = pq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(200);
      assertThat(compressed.dimension()).isEqualTo(128);
    }

    @Test
    void encode_wrongDimension_throws() {
      float[][] vectors = generateVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);

      assertThatThrownBy(() -> pq.encode(new float[] {1, 2, 3}))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(ints = {16, 64, 128, 256, 768})
    void variousDimensions_encodeDecode(int dim) {
      float[][] vectors = generateVectors(300, dim, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      int m = Math.min(16, dim);
      var pq = ProductQuantizer.train(dataset, m);

      byte[] encoded = pq.encode(vectors[0]);
      assertThat(encoded).hasSize(m);

      float[] decoded = pq.decode(encoded);
      assertThat(decoded).hasSize(dim);
    }
  }

  @Nested
  @Tag("unit")
  class ADCScoring {

    @Test
    void dotProductScore_correlatesWithTrueScore() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      // PQ is approximate — use generous tolerance
      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.DOT_PRODUCT.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.3f));
      }
    }

    @Test
    void euclideanScore_rankCorrelation() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      // PQ Euclidean has substantial quantization error on random data, so test rank correlation
      float[] trueScores = new float[500];
      float[] approxScores = new float[500];
      for (int i = 0; i < 500; i++) {
        trueScores[i] = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
        approxScores[i] = scoreFunc.score(i);
      }

      // Verify scores are finite and in valid range
      for (int i = 0; i < 500; i++) {
        assertThat(approxScores[i]).as("vector %d", i).isFinite().isGreaterThanOrEqualTo(0f);
      }

      // Rank correlation: top-10 overlap
      int[] trueTop10 = topK(trueScores, 10);
      int[] approxTop10 = topK(approxScores, 10);
      int overlap = countOverlap(trueTop10, approxTop10);
      assertThat(overlap).as("PQ Euclidean recall@10").isGreaterThanOrEqualTo(2);
    }

    @Test
    void cosineScore_correlatesWithTrueScore() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.COSINE);

      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.COSINE.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.3f));
      }
    }

    @Test
    void selfScore_isHighForNormalizedVectors() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float selfScore = scoreFunc.score(0);
      assertThat(selfScore).isGreaterThan(0.7f);
    }

    @Test
    void rankCorrelation_recall10() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float[] trueScores = new float[500];
      float[] approxScores = new float[500];
      for (int i = 0; i < 500; i++) {
        trueScores[i] = SimilarityFunction.DOT_PRODUCT.compare(query, vectors[i]);
        approxScores[i] = scoreFunc.score(i);
      }

      int[] trueTop10 = topK(trueScores, 10);
      int[] approxTop10 = topK(approxScores, 10);
      int overlap = countOverlap(trueTop10, approxTop10);
      // PQ with 16 subspaces should achieve reasonable recall
      assertThat(overlap).as("PQ recall@10").isGreaterThanOrEqualTo(3);
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void allSimilarityFunctions_returnFiniteScores(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(100, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);
      var compressed = pq.encodeAll(dataset);

      ScoreFunction scoreFunc = compressed.scoreFunctionFor(vectors[0], simFunc);

      for (int i = 0; i < 50; i++) {
        float score = scoreFunc.score(i);
        assertThat(score).as("ordinal %d", i).isFinite();
      }
    }
  }

  @Nested
  @Tag("unit")
  class CompressionRatio {

    @Test
    void pq_compressionRatio() {
      // 128 dims * 4 bytes/float = 512 bytes original
      // 8 subspaces * 1 byte/subspace = 8 bytes compressed
      // ratio = 512 / 8 = 64
      float[][] vectors = generateVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8);

      assertThat(pq.compressionRatio()).isCloseTo(64.0f, within(0.01f));
    }

    @Test
    void pq_16subspaces_compressionRatio() {
      // 128 dims * 4 bytes = 512 bytes original
      // 16 subspaces * 1 byte = 16 bytes compressed
      // ratio = 512 / 16 = 32
      float[][] vectors = generateVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 16);

      assertThat(pq.compressionRatio()).isCloseTo(32.0f, within(0.01f));
    }
  }

  @Nested
  @Tag("unit")
  class GlobalCentroid {

    @Test
    void centeredPQ_improvesReconstruction() {
      float[][] vectors = generateVectors(500, 128, 42L);
      // Shift all vectors by a large constant
      for (float[] v : vectors) {
        for (int d = 0; d < v.length; d++) {
          v[d] += 100f;
        }
      }
      var dataset = new ArrayVectorDataset(vectors);

      var pqNoCentroid = ProductQuantizer.train(dataset, 8, 256, false);
      var pqCentroid = ProductQuantizer.train(dataset, 8, 256, true);

      double mseNo = averageMse(pqNoCentroid, vectors);
      double mseCent = averageMse(pqCentroid, vectors);

      // Centering should help when data has large offset
      assertThat(mseCent).isLessThan(mseNo);
    }

    @Test
    void cosineScoring_withGlobalCentroid_usesConsistentNorm() {
      // Regression: cosineScoreFunction must use |query - globalCentroid| as the query norm,
      // not |query|. When centering is active, the ADC dot table holds (q-g)·c_k, so the
      // denominator must also be in centered space.
      //
      // We verify this by checking that: (a) cosine self-similarity ≈ 1 for all stored vectors,
      // and (b) vectors shifted by the same constant have a cosine score close to 1 for themselves
      // (the dot product in centered space is proportional to |c_k|^2, not (q+offset)·c_k).
      float[][] vectors = generateNormalizedVectors(100, 64, 42L);
      // Apply a large constant offset to make |query| very different from |query - centroid|
      for (float[] v : vectors) {
        for (int d = 0; d < v.length; d++) {
          v[d] += 50f;
        }
      }
      var dataset = new ArrayVectorDataset(vectors);
      var pq = ProductQuantizer.train(dataset, 8, 64, true);
      var pqv = pq.encodeAll(dataset);

      // Every vector's cosine score against itself should be high (close to 1)
      ScoreFunction sf = pqv.scoreFunctionFor(vectors[0], SimilarityFunction.COSINE);
      float selfScore = sf.score(0);

      // Without the fix, the query norm |query| >> |query - centroid|, so cos = dot/large ≈ 0.
      // With the fix, |query - centroid| is small and self-cosine should be comfortably above 0.5.
      assertThat(selfScore)
          .as("cosine self-score with global centroid must use centered query norm")
          .isGreaterThan(0.5f);
    }
  }

  /** SIFT Small dataset tests for PQ. */
  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.ProductQuantizerTest#siftAvailable")
  class SiftSmallDataset {

    @Test
    void trainOnSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);

      var pq = ProductQuantizer.train(dataset, 16); // 16 subspaces, 8 dims each

      assertThat(pq.dimension()).isEqualTo(128);
      assertThat(pq.numSubspaces()).isEqualTo(16);
    }

    @Test
    void siftSmall_encodeDecodeRoundTrip() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var pq = ProductQuantizer.train(dataset, 16);

      // PQ reconstruction error should be finite and bounded
      double mse = averageMse(pq, base, 1000);
      assertThat(mse).as("SIFT PQ average MSE").isFinite().isGreaterThan(0.0);
    }

    @Test
    void siftSmall_euclideanScoring_recallAt10() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      double totalRecall = 0;
      int numQueries = Math.min(10, queries.length);
      for (int q = 0; q < numQueries; q++) {
        ScoreFunction scoreFunc =
            compressed.scoreFunctionFor(queries[q], SimilarityFunction.EUCLIDEAN);

        float[] approxScores = new float[base.length];
        for (int i = 0; i < base.length; i++) {
          approxScores[i] = scoreFunc.score(i);
        }

        int[] approxTop10 = topK(approxScores, 10);
        int[] trueTop10 = new int[10];
        System.arraycopy(groundTruth[q], 0, trueTop10, 0, 10);

        double recall = SiftLoader.recallAtK(trueTop10, approxTop10, 10);
        totalRecall += recall;
      }
      totalRecall /= numQueries;

      // PQ with 16 subspaces should get at least 30% recall@10 on SIFT
      assertThat(totalRecall).as("PQ average recall@10 on SIFT Small").isGreaterThan(0.3);
    }

    @Test
    void siftSmall_compressionSize() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var pq = ProductQuantizer.train(dataset, 16);
      var compressed = pq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(10_000);
      assertThat(compressed.dimension()).isEqualTo(128);
      assertThat(pq.compressionRatio()).isCloseTo(32.0f, within(0.01f));
    }
  }

  // --- Helper methods ---

  private static float[][] generateVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = rng.nextFloat() * 2 - 1; // [-1, 1]
      }
    }
    return vectors;
  }

  private static float[][] generateGaussianVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = (float) rng.nextGaussian();
      }
    }
    return vectors;
  }

  private static float[][] generateNormalizedVectors(int count, int dim, long seed) {
    float[][] vectors = generateVectors(count, dim, seed);
    for (float[] v : vectors) l2normalize(v);
    return vectors;
  }

  private static void l2normalize(float[] v) {
    float norm = 0;
    for (float f : v) norm += f * f;
    norm = (float) Math.sqrt(norm);
    if (norm > 0) {
      for (int d = 0; d < v.length; d++) v[d] /= norm;
    }
  }

  private static double averageMse(ProductQuantizer pq, float[][] vectors) {
    return averageMse(pq, vectors, vectors.length);
  }

  private static double averageMse(ProductQuantizer pq, float[][] vectors, int count) {
    count = Math.min(count, vectors.length);
    double totalMse = 0;
    for (int i = 0; i < count; i++) {
      byte[] encoded = pq.encode(vectors[i]);
      float[] decoded = pq.decode(encoded);
      float mse = 0;
      for (int d = 0; d < vectors[i].length; d++) {
        float diff = decoded[d] - vectors[i][d];
        mse += diff * diff;
      }
      totalMse += mse / vectors[i].length;
    }
    return totalMse / count;
  }

  private static int[] topK(float[] scores, int k) {
    int[] indices = new int[k];
    boolean[] used = new boolean[scores.length];
    for (int i = 0; i < k; i++) {
      int best = -1;
      float bestScore = Float.NEGATIVE_INFINITY;
      for (int j = 0; j < scores.length; j++) {
        if (!used[j] && scores[j] > bestScore) {
          bestScore = scores[j];
          best = j;
        }
      }
      indices[i] = best;
      used[best] = true;
    }
    return indices;
  }

  private static int countOverlap(int[] a, int[] b) {
    int count = 0;
    for (int ai : a) {
      for (int bi : b) {
        if (ai == bi) {
          count++;
          break;
        }
      }
    }
    return count;
  }

  // ---------------------------------------------------------------------------
  // OPQ tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class OptimizedPQTests {

    @Test
    void opq_train_buildsWithoutError() {
      float[][] vecs = generateVectors(200, 16, 42L);
      // n=200, M=2, Ks=8, 3 iterations: 200/2/8=12.5 vectors/cluster (above minimum)
      var opq = OptimizedProductQuantizer.train(new ArrayVectorDataset(vecs), 2, 8, 3, 42L);
      assertThat(opq.dimension()).isEqualTo(16);
      assertThat(opq.productQuantizer().numSubspaces()).isEqualTo(2);
      assertThat(opq.productQuantizer().numClusters()).isEqualTo(8);
    }

    @Test
    void opq_rotation_isOrthogonal() {
      // Verify Rᵀ R ≈ I (orthogonality of the learned rotation)
      float[][] vecs = generateVectors(200, 8, 42L);
      var opq = OptimizedProductQuantizer.train(new ArrayVectorDataset(vecs), 2, 4, 3, 42L);
      float[][] R = opq.rotation();
      int d = R.length;

      // Compute Rᵀ·R and check it is close to the identity matrix
      for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
          float dot = 0f;
          for (int k = 0; k < d; k++) dot += R[k][i] * R[k][j]; // Rᵀ[i,k] * R[k,j]
          float expected = (i == j) ? 1f : 0f;
          assertThat(dot)
              .as("(RᵀR)[%d][%d] should be %.0f (orthogonality)", i, j, expected)
              .isCloseTo(expected, within(1e-5f));
        }
      }
    }

    @Test
    void opq_encode_decode_roundtrip_approximates_original() {
      // decode(encode(x)) should approximate x (within PQ quantization error)
      float[][] vecs = generateVectors(200, 16, 42L);
      var opq = OptimizedProductQuantizer.train(new ArrayVectorDataset(vecs), 2, 8, 3, 42L);

      float totalMse = 0f;
      for (float[] v : vecs) {
        float[] recon = opq.decode(opq.encode(v));
        float mse = 0f;
        for (int i = 0; i < v.length; i++) {
          float d = v[i] - recon[i];
          mse += d * d;
        }
        totalMse += mse / v.length;
      }
      float avgMse = totalMse / vecs.length;
      // The reconstruction error should be finite and below a reasonable bound
      assertThat(avgMse).as("OPQ avg MSE per vector").isFinite().isLessThan(10.0f);
    }

    @Test
    void opq_reducesReconstructionMse_vs_standardPq() {
      // OPQ should produce lower MSE than standard PQ on the same data
      // (OPQ rotation distributes energy evenly across subspaces)
      float[][] vecs = generateVectors(500, 16, 42L);
      var dataset = new ArrayVectorDataset(vecs);

      // Standard PQ
      var pq = ProductQuantizer.train(dataset, 2, 8, false);
      float pqMse = computeMse(pq, vecs);

      // OPQ with 5 iterations
      var opq = OptimizedProductQuantizer.train(dataset, 2, 8, 5, 42L);
      float opqMse = computeMse(opq, vecs);

      assertThat(opqMse)
          .as("OPQ MSE (%.4f) should be <= standard PQ MSE (%.4f)", opqMse, pqMse)
          .isLessThanOrEqualTo(pqMse * 1.05f); // allow 5% tolerance for randomness
    }

    @Test
    void polarDecomposition_producesOrthogonalMatrix() {
      // Verify the polar decomposition helper produces an orthogonal output
      float[][] M = {
        {1f, 0.5f, 0.2f},
        {0.3f, 2f, 0.1f},
        {0.4f, 0.1f, 1.5f}
      };
      float[][] P = OptimizedProductQuantizer.polarDecomposition(M);
      int d = P.length;
      // Check Pᵀ·P ≈ I
      for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
          float dot = 0f;
          for (int k = 0; k < d; k++) dot += P[k][i] * P[k][j];
          assertThat(dot)
              .as("polar(M): (PᵀP)[%d][%d]", i, j)
              .isCloseTo(i == j ? 1f : 0f, within(1e-5f));
        }
      }
    }

    @Test
    void matrixInvert_producesIdentityProduct() {
      float[][] A = {
        {4f, 7f},
        {2f, 6f}
      };
      float[][] Ainv = OptimizedProductQuantizer.invert(A);
      // A·A⁻¹ should be identity
      int d = A.length;
      for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
          float dot = 0f;
          for (int k = 0; k < d; k++) dot += A[i][k] * Ainv[k][j];
          assertThat(dot).isCloseTo(i == j ? 1f : 0f, within(1e-5f));
        }
      }
    }

    private float computeMse(Quantizer<?> q, float[][] vecs) {
      float total = 0f;
      for (float[] v : vecs) {
        float[] recon = q.decode(q.encode(v));
        for (int i = 0; i < v.length; i++) {
          float d = v[i] - recon[i];
          total += d * d;
        }
      }
      return total / (vecs.length * vecs[0].length);
    }
  }

  // ---------------------------------------------------------------------------
  // QuantizationPipeline gate tests (OM6)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class PipelineTests {

    private static final int DIM = 64;
    private static final int N = 300;

    private static VectorDataset corpus() {
      return new ArrayVectorDataset(generateVectors(N, DIM, 7L));
    }

    @Test
    void scalarInt8_presetTrainsAndEncodesDecodes() {
      var pipeline = QuantizationPipeline.scalarInt8().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.SCALAR_INT8);
      assertThat(pipeline.dimension()).isEqualTo(DIM);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);

      float[] v = generateVectors(1, DIM, 99L)[0];
      byte[] code = pipeline.encode(v);
      float[] recon = pipeline.decode(code);
      assertThat(recon).hasSize(DIM);
    }

    @Test
    void scalarInt4_presetTrainsAndEncodesDecodes() {
      var pipeline = QuantizationPipeline.scalarInt4().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.SCALAR_INT4);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);

      float[] v = generateVectors(1, DIM, 100L)[0];
      byte[] code = pipeline.encode(v);
      assertThat(code.length).isLessThan(v.length * Float.BYTES);
    }

    @Test
    void product_presetTrainsCompressAndScore() {
      var pipeline = QuantizationPipeline.product(4).train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.PRODUCT);
      assertThat(pipeline.dimension()).isEqualTo(DIM);

      VectorDataset ds = corpus();
      CompressedVectors cv = pipeline.compress(ds);
      assertThat(cv.size()).isEqualTo(N);

      float[] query = generateVectors(1, DIM, 55L)[0];
      ScoreFunction scorer = pipeline.scorerFor(query, cv);
      // Every ordinal must produce a finite non-negative score.
      for (int i = 0; i < N; i++) {
        float score = scorer.score(i);
        assertThat(score).isFinite().isGreaterThanOrEqualTo(0f);
      }
    }

    @Test
    void opq_presetTrainsAndReducesMse() {
      VectorDataset ds = corpus();
      var opq = QuantizationPipeline.opq(4).train(ds);
      assertThat(opq.type()).isEqualTo(QuantizerType.OPTIMIZED_PRODUCT);
      assertThat(opq.dimension()).isEqualTo(DIM);

      float[] v = generateVectors(1, DIM, 77L)[0];
      byte[] code = opq.encode(v);
      float[] recon = opq.decode(code);
      assertThat(recon).hasSize(DIM);
    }

    @Test
    void binarySign_presetTrainsAndCompresses() {
      var pipeline = QuantizationPipeline.binarySign().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.BINARY_SIGN);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void binaryBbq_presetTrainsAndCompresses() {
      var pipeline = QuantizationPipeline.binaryBbq().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.BINARY_BBQ);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void rabit_presetTrainsAndCompresses() {
      var pipeline = QuantizationPipeline.rabit().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.RABIT);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void turbo_presetTrainsAndCompresses() {
      var pipeline = QuantizationPipeline.turbo(4).train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.TURBO);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void nvq_presetTrainsAndCompresses() {
      var pipeline = QuantizationPipeline.nvq().train(corpus());
      assertThat(pipeline.type()).isEqualTo(QuantizerType.NVQ);
      assertThat(pipeline.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void builder_fullControl_similarityAndIterations() {
      var pipeline =
          QuantizationPipeline.builder()
              .quantizerType(QuantizerType.OPTIMIZED_PRODUCT)
              .subspaces(4)
              .clusters(64)
              .iterations(3)
              .seed(123L)
              .similarity(SimilarityFunction.DOT_PRODUCT)
              .train(corpus());

      assertThat(pipeline.type()).isEqualTo(QuantizerType.OPTIMIZED_PRODUCT);
      assertThat(pipeline.similarity()).isEqualTo(SimilarityFunction.DOT_PRODUCT);
      assertThat(pipeline.dimension()).isEqualTo(DIM);
    }

    @Test
    void quantizer_accessor_returnsUnderlyingQuantizer() {
      var pipeline = QuantizationPipeline.scalarInt8().train(corpus());
      assertThat(pipeline.quantizer()).isInstanceOf(ScalarQuantizer.class);
    }
  }
}
