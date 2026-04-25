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

/** Tests for {@link ScalarQuantizer} and {@link ScalarQuantizedVectors}. */
class ScalarQuantizerTest {

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
    void trainFromDataset_computesReasonableQuantiles() {
      float[][] vectors = generateVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);

      // Data is uniform [-1, 1], so quantiles should be close to those bounds
      assertThat(sq.minQuantile()).isLessThan(0f);
      assertThat(sq.maxQuantile()).isGreaterThan(0f);
      assertThat(sq.dimension()).isEqualTo(128);
      assertThat(sq.alpha()).isGreaterThan(0f);
      assertThat(sq.scale()).isGreaterThan(0f);
    }

    @Test
    void trainWithFullConfidence_usesAbsoluteMinMax() {
      float[][] vectors = {{-2f, -1f, 0f, 1f}, {-0.5f, 0.5f, 1.5f, 2f}};
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 1.0f);

      assertThat(sq.minQuantile()).isEqualTo(-2f);
      assertThat(sq.maxQuantile()).isEqualTo(2f);
    }

    @Test
    void trainWithNarrowConfidence_excludesOutliers() {
      Random rng = new Random(42L);
      float[][] vectors = new float[1000][16];
      for (int i = 0; i < 1000; i++) {
        for (int d = 0; d < 16; d++) {
          vectors[i][d] = (float) rng.nextGaussian() * 0.1f;
        }
      }
      // Add outliers
      vectors[0][0] = 10f;
      vectors[1][0] = -10f;

      var dataset = new ArrayVectorDataset(vectors);
      var sqFull = ScalarQuantizer.train(dataset, 1.0f);
      var sqNarrow = ScalarQuantizer.train(dataset, 0.99f);

      assertThat(sqNarrow.maxQuantile()).isLessThan(sqFull.maxQuantile());
      assertThat(sqNarrow.minQuantile()).isGreaterThan(sqFull.minQuantile());
    }

    /** Adapted from Lucene TestScalarQuantizer: quantile percentile verification. */
    @Test
    void quantileComputation_matchesExpectedPercentiles() {
      // Create a dataset with known distribution: 1000 values from 0 to 999
      float[][] vectors = new float[1000][1];
      for (int i = 0; i < 1000; i++) {
        vectors[i][0] = (float) i;
      }
      var dataset = new ArrayVectorDataset(vectors);

      var sq90 = ScalarQuantizer.train(dataset, 0.90f);
      // With CI=0.90, skip bottom 5% and top 5%: ~50 and ~949
      assertThat(sq90.minQuantile()).isBetween(40f, 60f);
      assertThat(sq90.maxQuantile()).isBetween(939f, 960f);

      var sq95 = ScalarQuantizer.train(dataset, 0.95f);
      // With CI=0.95, skip bottom 2.5% and top 2.5%: ~25 and ~974
      assertThat(sq95.minQuantile()).isBetween(15f, 35f);
      assertThat(sq95.maxQuantile()).isBetween(964f, 985f);
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 16, 128, 768, 1536})
    void trainVariousDimensions(int dim) {
      float[][] vectors = generateVectors(100, dim, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);

      assertThat(sq.dimension()).isEqualTo(dim);
      assertThat(sq.minQuantile()).isLessThan(sq.maxQuantile());
    }

    @Test
    void fromQuantiles_setsExplicitBounds() {
      var sq = ScalarQuantizer.fromQuantiles(128, -1.5f, 1.5f);

      assertThat(sq.minQuantile()).isEqualTo(-1.5f);
      assertThat(sq.maxQuantile()).isEqualTo(1.5f);
      assertThat(sq.dimension()).isEqualTo(128);
      assertThat(sq.alpha()).isCloseTo(3.0f / 127f, within(1e-6f));
      assertThat(sq.scale()).isCloseTo(127f / 3.0f, within(1e-4f));
    }

    @Test
    void invalidConfidenceInterval_throws() {
      float[][] vectors = generateVectors(10, 4, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      assertThatThrownBy(() -> ScalarQuantizer.train(dataset, 0f))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> ScalarQuantizer.train(dataset, -0.5f))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> ScalarQuantizer.train(dataset, 1.5f))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void fromQuantiles_invalidRange_throws() {
      assertThatThrownBy(() -> ScalarQuantizer.fromQuantiles(4, 1f, 0f))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> ScalarQuantizer.fromQuantiles(4, 1f, 1f))
          .isInstanceOf(IllegalArgumentException.class);
    }

    /** Gaussian data (JVector pattern): tests realistic embedding distributions. */
    @Test
    void trainOnGaussianData() {
      float[][] vectors = generateGaussianVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 0.99f);

      // Gaussian data centered at 0, quantiles should be roughly symmetric
      assertThat(sq.minQuantile()).isLessThan(0f);
      assertThat(sq.maxQuantile()).isGreaterThan(0f);
      assertThat(Math.abs(sq.minQuantile() + sq.maxQuantile())).isLessThan(1f);
    }
  }

  @Nested
  @Tag("unit")
  class EncodeAndDecode {

    @Test
    void roundTrip_withinTolerance() {
      var sq = ScalarQuantizer.fromQuantiles(128, -1.0f, 1.0f);
      float[] vector = generateVectors(1, 128, 42L)[0];

      byte[] encoded = sq.encode(vector);
      float[] decoded = sq.decode(encoded);

      assertThat(decoded).hasSize(128);
      float maxError = sq.alpha(); // max quantization error per dimension
      for (int i = 0; i < 128; i++) {
        float clamped = Math.max(-1f, Math.min(1f, vector[i]));
        assertThat(decoded[i]).isCloseTo(clamped, within(maxError + 1e-6f));
      }
    }

    /**
     * Lucene pattern: MAE (Mean Absolute Error) should be bounded by 1 / (2^bits). For 7-bit
     * quantization, MAE <= 1/128 ≈ 0.0078.
     */
    @Test
    void meanAbsoluteError_boundedByPrecision() {
      int dims = 128;
      float[][] vectors = generateVectors(200, dims, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 1.0f);

      float totalMae = 0;
      for (float[] v : vectors) {
        byte[] encoded = sq.encode(v);
        float[] decoded = sq.decode(encoded);
        float mae = 0;
        for (int d = 0; d < dims; d++) {
          float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), v[d]));
          mae += Math.abs(decoded[d] - clamped);
        }
        mae /= dims;
        totalMae += mae;
      }
      totalMae /= vectors.length;

      // MAE should be less than alpha/2 (half the quantization step)
      float eps = sq.alpha() / 2f;
      assertThat(totalMae).as("average MAE").isLessThan(eps);
    }

    @Test
    void encodedBytesInRange_0to127() {
      var sq = ScalarQuantizer.fromQuantiles(4, -1.0f, 1.0f);
      float[] v = {-1.0f, 0.0f, 0.5f, 1.0f};
      byte[] encoded = sq.encode(v);

      for (byte b : encoded) {
        assertThat(b & 0xFF).isBetween(0, 127);
      }
      assertThat(encoded[0] & 0xFF).isEqualTo(0); // -1.0 maps to 0
      assertThat(encoded[3] & 0xFF).isEqualTo(127); // 1.0 maps to 127
    }

    @Test
    void encodeClamps_outOfRangeValues() {
      var sq = ScalarQuantizer.fromQuantiles(4, -1.0f, 1.0f);
      float[] v = {-5.0f, 5.0f, -1.0f, 1.0f};
      byte[] encoded = sq.encode(v);

      // Out-of-range values should be clamped to [0, 127]
      assertThat(encoded[0] & 0xFF).isEqualTo(0);
      assertThat(encoded[1] & 0xFF).isEqualTo(127);
    }

    @Test
    void encode_returnsCorrectionFactor() {
      var sq = ScalarQuantizer.fromQuantiles(128, -1.0f, 1.0f);
      float[] vector = generateVectors(1, 128, 42L)[0];
      byte[] dst = new byte[128];

      float correction = sq.encode(vector, dst);

      // Correction should be finite
      assertThat(correction).isFinite();
    }

    @Test
    void encode_wrongDimension_throws() {
      var sq = ScalarQuantizer.fromQuantiles(4, -1.0f, 1.0f);

      assertThatThrownBy(() -> sq.encode(new float[] {1, 2, 3}))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 16, 128, 768})
    void encodeAll_producesCorrectSize(int dim) {
      float[][] vectors = generateVectors(100, dim, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(100);
      assertThat(compressed.dimension()).isEqualTo(dim);
    }

    @Test
    void decode_atBoundaries() {
      var sq = ScalarQuantizer.fromQuantiles(3, 0f, 1.0f);
      byte[] encoded = {0, 63, 127};
      float[] decoded = sq.decode(encoded);

      assertThat(decoded[0]).isCloseTo(0f, within(1e-6f));
      assertThat(decoded[1]).isCloseTo(63f / 127f, within(1e-6f));
      assertThat(decoded[2]).isCloseTo(1f, within(1e-6f));
    }

    /** Gaussian data round-trip (JVector pattern). */
    @Test
    void gaussianData_roundTrip_withinTolerance() {
      float[][] vectors = generateGaussianVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 0.99f);

      for (float[] v : vectors) {
        byte[] encoded = sq.encode(v);
        float[] decoded = sq.decode(encoded);
        for (int d = 0; d < 128; d++) {
          float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), v[d]));
          assertThat(decoded[d]).isCloseTo(clamped, within(sq.alpha() + 1e-5f));
        }
      }
    }
  }

  @Nested
  @Tag("unit")
  class Scoring {

    @Test
    void dotProductScore_correlatesWithTrueScore() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.DOT_PRODUCT.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.1f));
      }
    }

    @Test
    void euclideanScore_correlatesWithTrueScore() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.15f));
      }
    }

    @Test
    void cosineScore_correlatesWithTrueScore() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.COSINE);

      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.COSINE.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.1f));
      }
    }

    @Test
    void maximumInnerProduct_correlatesWithTrueScore() {
      float[][] vectors = generateVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc =
          compressed.scoreFunctionFor(query, SimilarityFunction.MAXIMUM_INNER_PRODUCT);

      for (int i = 0; i < 50; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.5f));
      }
    }

    @Test
    void selfScore_isHighForNormalizedVectors() {
      float[][] vectors = generateNormalizedVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float selfScore = scoreFunc.score(0);
      assertThat(selfScore).isGreaterThan(0.9f);
    }

    @Test
    void rankCorrelation_recall10_isHigh() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

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
      assertThat(overlap).as("recall@10").isGreaterThanOrEqualTo(7);
    }

    /**
     * Adapted from Lucene TestScalarQuantizedVectorSimilarity: CI-based error bounds. Error scales
     * with (1 - confidenceInterval).
     */
    @ParameterizedTest
    @ValueSource(floats = {0.9f, 0.95f, 0.99f, 1.0f})
    void scoringError_scalesWithConfidenceInterval(float ci) {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ci);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      float maxError = Math.max((1f - ci) * 0.5f, 0.02f);
      for (int i = 0; i < 50; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
        assertThat(Math.abs(approxScore - trueScore))
            .as("ci=%.2f, vector %d", ci, i)
            .isLessThan(maxError);
      }
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void allSimilarityFunctions_returnFiniteScores(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(50, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      ScoreFunction scoreFunc = compressed.scoreFunctionFor(vectors[0], simFunc);

      for (int i = 0; i < 50; i++) {
        float score = scoreFunc.score(i);
        assertThat(score).as("ordinal %d", i).isFinite();
      }
    }

    /** Gaussian data scoring (JVector pattern). */
    @Test
    void gaussianVectors_scoreCorrelation() {
      float[][] vectors = generateGaussianVectors(300, 128, 42L);
      // L2-normalize for dot product scoring
      for (float[] v : vectors) l2normalize(v);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float selfScore = scoreFunc.score(0);
      assertThat(selfScore).as("self-score on Gaussian data").isGreaterThan(0.85f);
    }
  }

  @Nested
  @Tag("unit")
  class CompressionRatio {

    @Test
    void int8_compressionIs4x() {
      var sq = ScalarQuantizer.fromQuantiles(128, -1f, 1f);
      assertThat(sq.compressionRatio()).isEqualTo(4.0f);
    }

    @Test
    void int4_compressionIs8x() {
      var sq = ScalarQuantizer.fromQuantiles(128, ScalarBits.INT4, -1f, 1f);
      assertThat(sq.compressionRatio()).isEqualTo(8.0f);
    }
  }

  @Nested
  @Tag("unit")
  class EdgeCases {

    @Test
    void uniformData_degenerateRange() {
      float[][] vectors = new float[100][16];
      for (float[] v : vectors) {
        java.util.Arrays.fill(v, 0.5f);
      }
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 1.0f);

      assertThat(sq.minQuantile()).isLessThan(sq.maxQuantile());
      byte[] encoded = sq.encode(vectors[0]);
      float[] decoded = sq.decode(encoded);
      for (int d = 0; d < 16; d++) {
        assertThat(decoded[d]).isCloseTo(0.5f, within(0.01f));
      }
    }

    @Test
    void smallDataset_noSampling() {
      float[][] vectors = generateVectors(10, 4, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);

      assertThat(sq.dimension()).isEqualTo(4);
      var compressed = sq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(10);
    }

    @Test
    void singleVector_trains() {
      float[][] vectors = {{1f, 2f, 3f, 4f}};
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, 1.0f);

      assertThat(sq.dimension()).isEqualTo(4);
      byte[] encoded = sq.encode(vectors[0]);
      assertThat(encoded).hasSize(4);
    }

    @Test
    void highDimensional_768() {
      float[][] vectors = generateNormalizedVectors(200, 768, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
      float selfScore = scoreFunc.score(0);
      assertThat(selfScore).isGreaterThan(0.85f);
    }

    /**
     * Adapted from JVector TestReconstructionError: reconstruction error statistics should be
     * stable across random samples.
     */
    @Test
    void reconstructionError_stableAcrossSamples() {
      float[][] vectors1 = generateVectors(500, 128, 42L);
      float[][] vectors2 = generateVectors(500, 128, 99L);
      var dataset1 = new ArrayVectorDataset(vectors1);
      var dataset2 = new ArrayVectorDataset(vectors2);

      var sq1 = ScalarQuantizer.train(dataset1);
      var sq2 = ScalarQuantizer.train(dataset2);

      double avgError1 = averageReconstructionError(sq1, vectors1);
      double avgError2 = averageReconstructionError(sq2, vectors2);

      // Relative error ratio should be close to 1.0 (JVector uses 1.15 tolerance)
      double ratio = avgError1 / avgError2;
      assertThat(ratio).as("error ratio").isBetween(0.5, 2.0);
    }
  }

  /** Int4 (4-bit) scalar quantization tests. */
  @Nested
  @Tag("unit")
  class Int4Quantization {

    @Test
    void train_int4_setsCorrectBits() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);

      assertThat(sq.bits()).isEqualTo(ScalarBits.INT4);
      assertThat(sq.dimension()).isEqualTo(128);
      assertThat(sq.compressionRatio()).isEqualTo(8.0f);
    }

    @Test
    void fromQuantiles_int4() {
      var sq = ScalarQuantizer.fromQuantiles(128, ScalarBits.INT4, -1.0f, 1.0f);

      assertThat(sq.bits()).isEqualTo(ScalarBits.INT4);
      assertThat(sq.alpha()).isCloseTo(2.0f / 15f, within(1e-6f));
      assertThat(sq.scale()).isCloseTo(15f / 2.0f, within(1e-4f));
    }

    @Test
    void encodedSize_isHalfDimension() {
      var sq = ScalarQuantizer.fromQuantiles(128, ScalarBits.INT4, -1.0f, 1.0f);
      float[] vector = generateVectors(1, 128, 42L)[0];
      byte[] encoded = sq.encode(vector);

      // Int4 packs 2 values per byte: ceil(128/2) = 64 bytes
      assertThat(encoded).hasSize(64);
    }

    @Test
    void encodedSize_oddDimension() {
      var sq = ScalarQuantizer.fromQuantiles(129, ScalarBits.INT4, -1.0f, 1.0f);
      float[] vector = generateVectors(1, 129, 42L)[0];
      byte[] encoded = sq.encode(vector);

      // ceil(129/2) = 65 bytes
      assertThat(encoded).hasSize(65);
    }

    @Test
    void packedNibbles_correctBitLayout() {
      var sq = ScalarQuantizer.fromQuantiles(4, ScalarBits.INT4, 0f, 15f);
      // Values exactly at quantized levels: 0, 5, 10, 15
      float[] vector = {0f, 5f, 10f, 15f};
      byte[] encoded = sq.encode(vector);

      // 4 dims -> 2 bytes
      assertThat(encoded).hasSize(2);
      // Byte 0: low nibble = dim 0 (value 0), high nibble = dim 1 (value 5)
      assertThat(encoded[0] & 0x0F).isEqualTo(0);
      assertThat((encoded[0] >> 4) & 0x0F).isEqualTo(5);
      // Byte 1: low nibble = dim 2 (value 10), high nibble = dim 3 (value 15)
      assertThat(encoded[1] & 0x0F).isEqualTo(10);
      assertThat((encoded[1] >> 4) & 0x0F).isEqualTo(15);
    }

    @Test
    void roundTrip_withinTolerance() {
      var sq = ScalarQuantizer.fromQuantiles(128, ScalarBits.INT4, -1.0f, 1.0f);
      float[] vector = generateVectors(1, 128, 42L)[0];

      byte[] encoded = sq.encode(vector);
      float[] decoded = sq.decode(encoded);

      assertThat(decoded).hasSize(128);
      float maxError = sq.alpha(); // 2.0/15 ≈ 0.133
      for (int i = 0; i < 128; i++) {
        float clamped = Math.max(-1f, Math.min(1f, vector[i]));
        assertThat(decoded[i]).as("dim %d", i).isCloseTo(clamped, within(maxError + 1e-5f));
      }
    }

    @Test
    void meanAbsoluteError_higherThanInt8() {
      int dims = 128;
      float[][] vectors = generateVectors(200, dims, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var sq8 = ScalarQuantizer.train(dataset, ScalarBits.INT8, 1.0f);
      var sq4 = ScalarQuantizer.train(dataset, ScalarBits.INT4, 1.0f);

      double mae8 = averageMae(sq8, vectors);
      double mae4 = averageMae(sq4, vectors);

      // Int4 has fewer levels, so higher MAE
      assertThat(mae4).isGreaterThan(mae8);
      // But int4 MAE should still be bounded by alpha/2
      assertThat(mae4).isLessThan(sq4.alpha() / 2f);
    }

    @Test
    void encodeAll_producesCorrectSize() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(100);
      assertThat(compressed.dimension()).isEqualTo(128);
      // Each stored vector is packed: 64 bytes
      assertThat(compressed.getQuantizedVector(0)).hasSize(64);
    }

    @Test
    void dotProductScore_correlatesWithTrueScore() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      // Int4 has wider tolerance than int8 due to coarser quantization
      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.DOT_PRODUCT.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.2f));
      }
    }

    @Test
    void euclideanScore_correlatesWithTrueScore() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

      for (int i = 0; i < 100; i++) {
        float approxScore = scoreFunc.score(i);
        float trueScore = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
        assertThat(approxScore).as("vector %d", i).isCloseTo(trueScore, within(0.25f));
      }
    }

    @Test
    void selfScore_isHighForNormalizedVectors() {
      float[][] vectors = generateNormalizedVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float selfScore = scoreFunc.score(0);
      // Int4 self-score is lower than int8 due to quantization noise
      assertThat(selfScore).isGreaterThan(0.8f);
    }

    @Test
    void rankCorrelation_recall10() {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

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
      // Int4 recall is lower than int8 but still meaningful
      assertThat(overlap).as("int4 recall@10").isGreaterThanOrEqualTo(5);
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void allSimilarityFunctions_returnFiniteScores(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(50, 64, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      ScoreFunction scoreFunc = compressed.scoreFunctionFor(vectors[0], simFunc);

      for (int i = 0; i < 50; i++) {
        float score = scoreFunc.score(i);
        assertThat(score).as("ordinal %d", i).isFinite();
      }
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 4, 15, 16, 17, 128, 129, 768})
    void variousDimensions_encodeDecodeRoundTrip(int dim) {
      float[][] vectors = generateVectors(50, dim, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);

      byte[] encoded = sq.encode(vectors[0]);
      assertThat(encoded).hasSize((dim + 1) / 2);

      float[] decoded = sq.decode(encoded);
      assertThat(decoded).hasSize(dim);

      for (int d = 0; d < dim; d++) {
        float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), vectors[0][d]));
        assertThat(decoded[d]).as("dim %d", d).isCloseTo(clamped, within(sq.alpha() + 1e-5f));
      }
    }

    /** Gaussian data (JVector pattern) with int4. */
    @Test
    void gaussianVectors_scoreCorrelation() {
      float[][] vectors = generateGaussianVectors(300, 128, 42L);
      for (float[] v : vectors) l2normalize(v);
      var dataset = new ArrayVectorDataset(vectors);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction scoreFunc = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);

      float selfScore = scoreFunc.score(0);
      assertThat(selfScore).as("int4 self-score on Gaussian data").isGreaterThan(0.75f);
    }

    @Test
    void encodeInt4_reusedDstBuffer_doesNotCorruptHighNibble() {
      // Regression: encodeInt4 OR-sets high nibbles, so a reused (dirty) buffer corrupts encoding
      // unless the buffer is zero-initialised first. Encode two different vectors into the same
      // byte[] and verify the second encoding is identical to a fresh encode.
      float[][] vecs = generateVectors(2, 64, 99L);
      var dataset = new ArrayVectorDataset(vecs);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);

      // First encode: fills the buffer with vector 0's nibbles
      byte[] reused = sq.encode(vecs[0]);

      // Second encode: must overwrite ALL nibbles cleanly (return value is correction factor,
      // unused here)
      sq.encode(vecs[1], reused);

      // Reference: fresh buffer encode of vector 1
      byte[] fresh = sq.encode(vecs[1]);

      assertThat(reused)
          .as("reused dst must match a fresh encode of vector 1 (high-nibble corruption check)")
          .isEqualTo(fresh);
    }
  }

  /** Int4 tests against SIFT Small dataset. */
  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.ScalarQuantizerTest#siftAvailable")
  class SiftSmallInt4 {

    @Test
    void siftSmall_int4_encodeDecodeRoundTrip() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4, 1.0f);

      float totalMae = 0;
      int sampleCount = Math.min(1000, base.length);
      for (int i = 0; i < sampleCount; i++) {
        byte[] encoded = sq.encode(base[i]);
        float[] decoded = sq.decode(encoded);
        float mae = 0;
        for (int d = 0; d < 128; d++) {
          float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), base[i][d]));
          mae += Math.abs(decoded[d] - clamped);
        }
        totalMae += mae / 128f;
      }
      totalMae /= sampleCount;

      // Int4 MAE should be less than alpha/2 but higher than int8
      assertThat(totalMae).as("SIFT int4 average MAE").isLessThan(sq.alpha() / 2f);
    }

    @Test
    void siftSmall_int4_euclideanScoring_recallAt10() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

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

      // Int4 recall is lower than int8 (~70%) but should still be meaningful (~40%+)
      assertThat(totalRecall).as("int4 average recall@10 on SIFT Small").isGreaterThan(0.4);
    }

    @Test
    void siftSmall_int4_compressionSize() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset, ScalarBits.INT4);
      var compressed = sq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(10_000);
      assertThat(compressed.dimension()).isEqualTo(128);
      assertThat(sq.compressionRatio()).isEqualTo(8.0f);
      // Each stored vector: 128 dims / 2 = 64 bytes (packed)
      assertThat(compressed.getQuantizedVector(0)).hasSize(64);
    }
  }

  /** Tests against the real SIFT Small dataset (10,000 vectors, 128 dimensions). */
  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.ScalarQuantizerTest#siftAvailable")
  class SiftSmallDataset {

    @Test
    void trainOnSiftSmall() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      assertThat(base.length).isEqualTo(10_000);
      assertThat(base[0].length).isEqualTo(128);

      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset);

      assertThat(sq.dimension()).isEqualTo(128);
      assertThat(sq.minQuantile()).isGreaterThanOrEqualTo(0f); // SIFT vectors are non-negative
      assertThat(sq.maxQuantile()).isGreaterThan(0f);
    }

    @Test
    void siftSmall_encodeDecodeRoundTrip() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset, 1.0f);

      // Lucene-style MAE check: should be less than alpha/2
      float totalMae = 0;
      int sampleCount = Math.min(1000, base.length);
      for (int i = 0; i < sampleCount; i++) {
        byte[] encoded = sq.encode(base[i]);
        float[] decoded = sq.decode(encoded);
        float mae = 0;
        for (int d = 0; d < 128; d++) {
          float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), base[i][d]));
          mae += Math.abs(decoded[d] - clamped);
        }
        totalMae += mae / 128f;
      }
      totalMae /= sampleCount;

      assertThat(totalMae).as("SIFT average MAE").isLessThan(sq.alpha() / 2f);
    }

    @Test
    void siftSmall_euclideanScoring_recallAt10() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      // Test recall@10 over 10 queries
      double totalRecall = 0;
      int numQueries = Math.min(10, queries.length);
      for (int q = 0; q < numQueries; q++) {
        ScoreFunction scoreFunc =
            compressed.scoreFunctionFor(queries[q], SimilarityFunction.EUCLIDEAN);

        // Compute approximate scores for all base vectors
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

      // SQ int8 with oversampling=1x should get at least 70% recall on SIFT
      assertThat(totalRecall).as("average recall@10 on SIFT Small").isGreaterThan(0.7);
    }

    @Test
    void siftSmall_compressionSize() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      var dataset = SiftLoader.asDataset(base);
      var sq = ScalarQuantizer.train(dataset);
      var compressed = sq.encodeAll(dataset);

      assertThat(compressed.size()).isEqualTo(10_000);
      assertThat(compressed.dimension()).isEqualTo(128);
      assertThat(sq.compressionRatio()).isEqualTo(4.0f);
    }
  }

  @Nested
  @Tag("unit")
  class ArrayVectorDatasetTest {

    @Test
    void basicProperties() {
      float[][] vectors = {{1, 2, 3}, {4, 5, 6}};
      var dataset = new ArrayVectorDataset(vectors);

      assertThat(dataset.size()).isEqualTo(2);
      assertThat(dataset.dimension()).isEqualTo(3);
      assertThat(dataset.getVector(0)).containsExactly(1f, 2f, 3f);
      assertThat(dataset.getVector(1)).containsExactly(4f, 5f, 6f);
    }

    @Test
    void emptyDataset_throws() {
      assertThatThrownBy(() -> new ArrayVectorDataset(new float[0][]))
          .isInstanceOf(IllegalArgumentException.class);
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

  /** Gaussian random vectors N(0, 1) per dimension (JVector pattern). */
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

  private static double averageMae(ScalarQuantizer sq, float[][] vectors) {
    double total = 0;
    for (float[] v : vectors) {
      byte[] encoded = sq.encode(v);
      float[] decoded = sq.decode(encoded);
      float mae = 0;
      for (int d = 0; d < v.length; d++) {
        float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), v[d]));
        mae += Math.abs(decoded[d] - clamped);
      }
      total += mae / v.length;
    }
    return total / vectors.length;
  }

  private static double averageReconstructionError(ScalarQuantizer sq, float[][] vectors) {
    double totalError = 0;
    for (float[] v : vectors) {
      byte[] encoded = sq.encode(v);
      float[] decoded = sq.decode(encoded);
      float error = 0;
      for (int d = 0; d < v.length; d++) {
        float clamped = Math.max(sq.minQuantile(), Math.min(sq.maxQuantile(), v[d]));
        float diff = decoded[d] - clamped;
        error += diff * diff;
      }
      totalError += error / v.length;
    }
    return totalError / vectors.length;
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
}
