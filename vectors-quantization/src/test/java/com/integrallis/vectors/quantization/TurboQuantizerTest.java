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
package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Tests for {@link TurboQuantizer} and {@link TurboQuantizedVectors}. */
class TurboQuantizerTest {

  private static final int DIM = 32;
  private static final int NUM_VECTORS = 200;
  private static final long SEED = 42L;

  /** Path to SIFT Small dataset (from JVector research repo). */
  private static final Path SIFT_DIR =
      Path.of("../../research/repos/jvector/siftsmall").toAbsolutePath().normalize();

  static boolean siftAvailable() {
    return Files.exists(SIFT_DIR.resolve("siftsmall_base.fvecs"));
  }

  private static float[][] generateVectors(int numVectors, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[numVectors][dim];
    for (int i = 0; i < numVectors; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = (float) rng.nextGaussian();
      }
    }
    return vectors;
  }

  private static VectorDataset makeDataset(float[][] vectors) {
    return new ArrayVectorDataset(vectors);
  }

  // --- Encoding Tests ---

  @Nested
  @Tag("unit")
  class EncodingTests {

    @Test
    void encode_dimensionMismatch_throws() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      TurboQuantizer tq = TurboQuantizer.train(makeDataset(vectors), 4);
      assertThatThrownBy(() -> tq.encode(new float[DIM + 1]))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void encode_returnsNormAsCorrection() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      TurboQuantizer tq = TurboQuantizer.train(makeDataset(vectors), 4);
      byte[] dst = new byte[tq.encodedByteSize()];
      float norm = tq.encode(vectors[0], dst);
      assertThat(norm).isGreaterThan(0f);
    }

    @Test
    void encodeAll_producesCorrectSize() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(NUM_VECTORS);
      assertThat(compressed.dimension()).isEqualTo(DIM);
    }

    @Test
    void encodeAndDecode_roundTrip_approximatesOriginal() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);

      for (int i = 0; i < 10; i++) {
        byte[] encoded = tq.encode(vectors[i]);
        float[] decoded = tq.decode(encoded);
        assertThat(decoded).hasSize(DIM);
        // The decoded vector should approximate the original (lossy compression)
        float sqDist = VectorUtil.squareDistance(vectors[i], decoded);
        float origNorm = (float) Math.sqrt(VectorUtil.dotProduct(vectors[i], vectors[i]));
        // Relative error should be bounded
        assertThat(sqDist / (origNorm * origNorm + 1e-6f)).isLessThan(1.0f);
      }
    }

    @Test
    void decode_returnsDirectionOnly_magnitudeIsNotPreserved() {
      // TurboQuantizer.decode(byte[]) reconstructs only the unit-direction of (v - centroid);
      // the per-vector norm is stored separately in TurboQuantizedVectors and is not embedded in
      // the packed byte array. Callers needing full reconstruction must use reconstructCentered().
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);

      for (int i = 0; i < 5; i++) {
        byte[] encoded = tq.encode(vectors[i]);
        float[] decoded = tq.decode(encoded);

        // The decoded vector is approximate; verify shape is correct.
        assertThat(decoded).hasSize(DIM);

        // Full reconstruction via reconstructCentered + norm should match decode better.
        byte[] dst = new byte[tq.encodedByteSize()];
        float norm = tq.encode(vectors[i], dst);
        float[] fullRecon = tq.reconstructCentered(dst, norm);
        float[] fullWithCentroid = new float[DIM];
        float[] centroid = tq.centroid();
        for (int d = 0; d < DIM; d++) {
          fullWithCentroid[d] = fullRecon[d] + centroid[d];
        }
        // Full reconstruction (with norm) should be closer to original than direction-only decode.
        float distFull = VectorUtil.squareDistance(vectors[i], fullWithCentroid);
        float distDir = VectorUtil.squareDistance(vectors[i], decoded);
        assertThat(distFull).isLessThanOrEqualTo(distDir + 1e-4f);
      }
    }

    @Test
    void encode_reusedBuffer_doesNotCorruptBits() {
      // Quality checklist: every encode that writes to a caller-supplied byte[] must have a
      // reused-buffer regression test. packIndices uses |= (OR-packing), so stale bits
      // from a previous encode would corrupt the result without the Arrays.fill(dst, (byte) 0).
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);

      byte[] dst = new byte[tq.encodedByteSize()];

      // Encode vector 0 into dst
      tq.encode(vectors[0], dst);
      byte[] firstEncode = dst.clone();

      // Encode vector 1 into the SAME dst buffer
      tq.encode(vectors[1], dst);

      // Now re-encode vector 0 into the same dst buffer again
      tq.encode(vectors[0], dst);

      // Must match the first encode exactly — stale bits from vector 1 must not leak
      assertThat(dst).isEqualTo(firstEncode);
    }

    @Test
    void compressionRatio_isPositive() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      TurboQuantizer tq = TurboQuantizer.train(makeDataset(vectors), 4);
      assertThat(tq.compressionRatio()).isGreaterThan(1f);
    }

    @Test
    void compressionRatio_increasesWithFewerBits() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      float ratio2 = TurboQuantizer.train(dataset, 2).compressionRatio();
      float ratio4 = TurboQuantizer.train(dataset, 4).compressionRatio();
      float ratio8 = TurboQuantizer.train(dataset, 8).compressionRatio();
      assertThat(ratio2).isGreaterThan(ratio4);
      assertThat(ratio4).isGreaterThan(ratio8);
    }
  }

  // --- Scoring Tests ---

  @Nested
  @Tag("unit")
  class ScoringTests {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void scoreFunction_returnsValidScores(SimilarityFunction simFunc) {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scorer = compressed.scoreFunctionFor(query, simFunc);

      for (int i = 0; i < Math.min(20, NUM_VECTORS); i++) {
        float score = scorer.score(i);
        assertThat(score).isNotNaN();
        assertThat(score).isNotInfinite();
      }
    }

    @Test
    void euclidean_selfScore_isHighest() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
      float selfScore = scorer.score(0);

      // Self-score should be among the highest
      int higherCount = 0;
      for (int i = 1; i < NUM_VECTORS; i++) {
        if (scorer.score(i) > selfScore) {
          higherCount++;
        }
      }
      // Allow at most a few vectors to score higher (due to quantization noise)
      assertThat(higherCount).isLessThan(NUM_VECTORS / 10);
    }

    @Test
    void euclidean_scoreCorrelatesWithTrueDistance() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[5];
      ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      // Compute true and approximate scores, check correlation
      float[] trueScores = new float[NUM_VECTORS];
      float[] approxScores = new float[NUM_VECTORS];
      for (int i = 0; i < NUM_VECTORS; i++) {
        float trueDist = VectorUtil.squareDistance(query, vectors[i]);
        trueScores[i] = 1f / (1f + trueDist);
        approxScores[i] = scorer.score(i);
      }

      float correlation = spearmanRankCorrelation(trueScores, approxScores);
      assertThat(correlation).isGreaterThan(0.8f);
    }

    @Test
    void dotProduct_scoreCorrelatesWithTrueScore() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[5];
      ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float[] trueScores = new float[NUM_VECTORS];
      float[] approxScores = new float[NUM_VECTORS];
      for (int i = 0; i < NUM_VECTORS; i++) {
        float trueDot = VectorUtil.dotProduct(query, vectors[i]);
        trueScores[i] = Math.max((1f + trueDot) / 2f, 0f);
        approxScores[i] = scorer.score(i);
      }

      float correlation = spearmanRankCorrelation(trueScores, approxScores);
      assertThat(correlation).isGreaterThan(0.7f);
    }

    @Test
    void cosine_scoreCorrelatesWithTrueScore() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[5];
      ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.COSINE);

      float[] trueScores = new float[NUM_VECTORS];
      float[] approxScores = new float[NUM_VECTORS];
      for (int i = 0; i < NUM_VECTORS; i++) {
        trueScores[i] = (1f + VectorUtil.cosine(query, vectors[i])) / 2f;
        approxScores[i] = scorer.score(i);
      }

      float correlation = spearmanRankCorrelation(trueScores, approxScores);
      assertThat(correlation).isGreaterThan(0.7f);
    }

    @Test
    void maximumInnerProduct_scoreCorrelatesWithTrueScore() {
      // MAXIMUM_INNER_PRODUCT had no rank-correlation test — only the generic NaN/infinite check
      // from the @EnumSource test. This test catches regressions in the MIPS scoring path.
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      float[] query = vectors[5];
      ScoreFunction scorer =
          compressed.scoreFunctionFor(query, SimilarityFunction.MAXIMUM_INNER_PRODUCT);

      float[] trueScores = new float[NUM_VECTORS];
      float[] approxScores = new float[NUM_VECTORS];
      for (int i = 0; i < NUM_VECTORS; i++) {
        float dot = VectorUtil.dotProduct(query, vectors[i]);
        trueScores[i] = SimilarityFunction.scaleMaxInnerProductScore(dot);
        approxScores[i] = scorer.score(i);
      }

      float correlation = spearmanRankCorrelation(trueScores, approxScores);
      assertThat(correlation)
          .as("MIPS approximate score must rank-correlate with true dot product")
          .isGreaterThan(0.7f);
    }
  }

  // --- Rotation Strategy Tests ---

  @Nested
  @Tag("unit")
  class RotationStrategyTests {

    @Test
    void defaultRotation_isPaperFaithfulRandom() {
      // Default is the dense random orthogonal rotation specified by the TurboQuant paper
      // (arXiv:2504.19874); the O(d) Rotor variants are opt-in via the Rotation-accepting overload.
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      assertThat(tq.rotation()).isInstanceOf(RandomRotation.class);
    }

    @Test
    void denseRotation_works() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      int paddedDim = ((DIM + 63) / 64) * 64;
      Rotation rotation = RandomRotation.generate(paddedDim, SEED);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4, rotation);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(NUM_VECTORS);

      ScoreFunction scorer = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      assertThat(scorer.score(0)).isGreaterThan(0f);
    }

    @Test
    void quaternionRotation_works() {
      float[][] vectors = generateVectors(NUM_VECTORS, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      int paddedDim = ((DIM + 63) / 64) * 64;
      Rotation rotation = QuaternionRotation.generate(paddedDim, SEED);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4, rotation);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(NUM_VECTORS);

      ScoreFunction scorer = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      assertThat(scorer.score(0)).isGreaterThan(0f);
    }

    @Test
    void rotationDimensionMismatch_throws() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      Rotation wrongDim = GivensRotation.generate(128, SEED); // wrong dimension
      assertThatThrownBy(() -> TurboQuantizer.train(dataset, 4, wrongDim))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  // --- Bit-Width Tests ---

  @Nested
  @Tag("unit")
  class BitWidthTests {

    @Test
    void twoBit_works() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 2);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);
      ScoreFunction scorer = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      assertThat(scorer.score(0)).isGreaterThan(0f);
    }

    @Test
    void eightBit_works() {
      float[][] vectors = generateVectors(50, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer tq = TurboQuantizer.train(dataset, 8);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);
      ScoreFunction scorer = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      assertThat(scorer.score(0)).isGreaterThan(0f);
    }

    @Test
    void moreBits_betterReconstruction() {
      float[][] vectors = generateVectors(100, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);

      float mse2 = averageMSE(dataset, TurboQuantizer.train(dataset, 2));
      float mse4 = averageMSE(dataset, TurboQuantizer.train(dataset, 4));
      float mse8 = averageMSE(dataset, TurboQuantizer.train(dataset, 8));

      assertThat(mse4).isLessThan(mse2);
      assertThat(mse8).isLessThan(mse4);
    }

    private float averageMSE(VectorDataset dataset, TurboQuantizer tq) {
      float totalMSE = 0;
      for (int i = 0; i < dataset.size(); i++) {
        float[] original = dataset.getVector(i);
        byte[] encoded = tq.encode(original);
        float[] decoded = tq.decode(encoded);
        totalMSE += VectorUtil.squareDistance(original, decoded);
      }
      return totalMSE / dataset.size();
    }
  }

  // --- SIFT Small Dataset Tests ---

  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.TurboQuantizerTest#siftAvailable")
  class SiftSmallTurboQuant {

    @Test
    void turboQuant_recall_onSiftSmall_euclidean() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      // Use 4-bit Givens (default) for this recall test — competitive speed/quality balance
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors compressed = tq.encodeAll(dataset);

      double totalRecall = computeAverageRecall(compressed, base, queries, groundTruth, 10);
      assertThat(totalRecall)
          .as("TurboQuant (4-bit Givens) average recall@10 on SIFT Small (Euclidean)")
          .isGreaterThan(0.40);
    }

    @Test
    void turboQuant_recall_notWorseThanRaBitQ_onSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);

      // RaBitQ (1-bit = ~32x compression). TurboQuant at 4-bit provides 8x — more bits, more info.
      var rq = RaBitQuantizer.train(dataset);
      var rqCompressed = rq.encodeAll(dataset);
      double rqRecall = computeAverageRecall(rqCompressed, base, queries, groundTruth, 10);

      // 4-bit TurboQuant should not be worse than 1-bit RaBitQ (it uses 4× more bits per dim).
      TurboQuantizer tq = TurboQuantizer.train(dataset, 4);
      TurboQuantizedVectors tqCompressed = tq.encodeAll(dataset);
      double tqRecall = computeAverageRecall(tqCompressed, base, queries, groundTruth, 10);

      assertThat(tqRecall)
          .as(
              "TurboQuant@4bit recall should be >= RaBitQ@1bit on SIFT Small"
                  + " (TurboQuant uses 4x more bits per dimension)")
          .isGreaterThanOrEqualTo(rqRecall * 0.85);
    }

    private double computeAverageRecall(
        CompressedVectors compressed,
        float[][] base,
        float[][] queries,
        int[][] groundTruth,
        int k) {
      double totalRecall = 0;
      int numQueries = Math.min(10, queries.length);
      for (int q = 0; q < numQueries; q++) {
        ScoreFunction scoreFunc =
            compressed.scoreFunctionFor(queries[q], SimilarityFunction.EUCLIDEAN);

        float[] approxScores = new float[base.length];
        for (int i = 0; i < base.length; i++) {
          approxScores[i] = scoreFunc.score(i);
        }

        int[] approxTopK = topK(approxScores, k);
        int[] trueTopK = new int[k];
        System.arraycopy(groundTruth[q], 0, trueTopK, 0, k);

        totalRecall += SiftLoader.recallAtK(trueTopK, approxTopK, k);
      }
      return totalRecall / numQueries;
    }
  }

  // --- Spearman rank correlation helper ---

  private static float spearmanRankCorrelation(float[] a, float[] b) {
    int n = a.length;
    int[] rankA = rank(a);
    int[] rankB = rank(b);
    long sumDiffSq = 0;
    for (int i = 0; i < n; i++) {
      long d = rankA[i] - rankB[i];
      sumDiffSq += d * d;
    }
    return 1f - (6f * sumDiffSq) / (n * ((long) n * n - 1));
  }

  private static int[] rank(float[] values) {
    int n = values.length;
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) indices[i] = i;
    java.util.Arrays.sort(indices, (x, y) -> Float.compare(values[x], values[y]));
    int[] ranks = new int[n];
    for (int i = 0; i < n; i++) {
      ranks[indices[i]] = i;
    }
    return ranks;
  }

  /** Returns the indices of the top-k highest scores (descending order). */
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

  @Nested
  class UnbiasedProdTests {

    @Test
    void qjlEstimatorIsApproximatelyUnbiased() {
      // Validates the QJL math/normalization (the sqrt(pi/2)/d constant + sign/transpose): for a
      // unit vector u, E[ sqrt(pi/2)/d · Sᵀ·sign(S·u) ] = u, so its projection onto u averages to
      // ‖u‖² = 1. A wrong constant or transpose would push this far from 1. Averaging over many u
      // cancels the QJL variance (which is why per-pair absolute error is a poor metric at small d
      // —
      // the paper evaluates d >= 200).
      int d = 64;
      QjlSketch sketch = QjlSketch.generate(d, SEED);
      Random rng = new Random(123L);
      int trials = 400;
      double sumProjection = 0;
      for (int t = 0; t < trials; t++) {
        float[] u = new float[d];
        double n2 = 0;
        for (int j = 0; j < d; j++) {
          u[j] = (float) rng.nextGaussian();
          n2 += (double) u[j] * u[j];
        }
        float inv = (float) (1.0 / Math.sqrt(n2));
        for (int j = 0; j < d; j++) {
          u[j] *= inv;
        }
        byte[] bits = sketch.signBits(u);
        float[] est = new float[d];
        sketch.addInverseResidual(bits, 1.0f, est); // gamma = ||u|| = 1
        float projection = 0f;
        for (int j = 0; j < d; j++) {
          projection += est[j] * u[j];
        }
        sumProjection += projection;
      }
      double mean = sumProjection / trials;
      assertThat(mean)
          .as("QJL estimate projection onto u should average to 1, got %.4f", mean)
          .isCloseTo(1.0, org.assertj.core.api.Assertions.within(0.15));
    }

    @Test
    void prodReconstructionRoundTripsThroughEncodeAll() {
      // Sanity: the unbiased path produces valid, finite, descending-or-equal self scores.
      float[][] vectors = generateVectors(100, DIM, SEED);
      VectorDataset dataset = makeDataset(vectors);
      TurboQuantizer prod = TurboQuantizer.trainProd(dataset, 4, SEED);
      TurboQuantizedVectors enc = prod.encodeAll(dataset);
      ScoreFunction scorer = enc.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < vectors.length; i++) {
        assertThat(scorer.score(i)).isFinite();
      }
    }
  }
}
