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
import com.integrallis.vectors.core.VectorUtil;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests for {@link ExtendedRaBitQuantizer} and {@link ExtendedRaBitQuantizedVectors}. Extended
 * RaBitQ (SIGMOD 2025) extends 1-bit RaBitQ to arbitrary bit-widths (2-8 bits/dim) using
 * correction-factor-based scoring (zero allocation per score).
 */
class ExtendedRaBitQuantizerTest {

  private static final Path SIFT_DIR =
      Path.of("../../research/repos/jvector/siftsmall").toAbsolutePath().normalize();

  static boolean siftAvailable() {
    return Files.exists(SIFT_DIR.resolve("siftsmall_base.fvecs"));
  }

  // --- Encoding Tests ---

  @Nested
  @Tag("unit")
  class EncodingTests {

    @Test
    void encode_centersNormalizesRotatesBeforeQuantize() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);

      byte[] encoded = eq.encode(vectors[0]);
      assertThat(encoded).hasSize(eq.encodedByteSize());

      // Sign bits portion should have a mix of 0 and 1 bits
      int signBytes = eq.signByteSize();
      int totalOnes = 0;
      for (int i = 0; i < signBytes; i++) {
        totalOnes += Integer.bitCount(encoded[i] & 0xFF);
      }
      int paddedDim = eq.paddedDimension();
      assertThat(totalOnes).isBetween(paddedDim / 4, 3 * paddedDim / 4);
    }

    @Test
    void encode_reusedBuffer_doesNotCorruptBits() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      byte[] dst = new byte[eq.encodedByteSize()];

      // Encode vector 0
      eq.encode(vectors[0], dst);
      byte[] first = java.util.Arrays.copyOf(dst, dst.length);

      // Encode vector 1 into same buffer
      eq.encode(vectors[1], dst);

      // Re-encode vector 0 and verify it matches first encoding
      eq.encode(vectors[0], dst);
      assertThat(dst).isEqualTo(first);
    }

    @Test
    void encode_dimensionMismatch_throws() {
      var eq = ExtendedRaBitQuantizer.train(makeDataset(128), 4);
      assertThatThrownBy(() -> eq.encode(new float[64]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Expected dimension 128");
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 9, 10, -1})
    void invalidBits_throws(int bits) {
      assertThatThrownBy(() -> ExtendedRaBitQuantizer.train(makeDataset(128), bits))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void encodeAndDecode_roundTrip_approximatesOriginal() {
      float[][] vectors = generateNormalizedVectors(50, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);

      for (int i = 0; i < 10; i++) {
        float[] decoded = eq.decode(eq.encode(vectors[i]));
        assertThat(decoded).hasSize(128);
        // 4-bit multi-bit preserves direction substantially better than 1-bit (which scores > 0.5).
        // Expected cosine for 4-bit is ~0.92+ on normalized Gaussian vectors; 0.85 is conservative.
        float cosine = VectorUtil.cosine(vectors[i], decoded);
        assertThat(cosine).as("Decoded vector cosine for vector %d", i).isGreaterThan(0.85f);
      }
    }

    @Test
    void encodedByteSize_matchesExpected() {
      // dim=128 → padded=128, 4-bit: signBytes=16, magBytes=(128*4+7)/8=64 → total=80
      var eq128_4 = ExtendedRaBitQuantizer.train(makeDataset(128), 4);
      assertThat(eq128_4.signByteSize()).isEqualTo(16);
      assertThat(eq128_4.magByteSize()).isEqualTo(64);
      assertThat(eq128_4.encodedByteSize()).isEqualTo(80);

      // dim=128, 2-bit: signBytes=16, magBytes=(128*2+7)/8=32 → total=48
      var eq128_2 = ExtendedRaBitQuantizer.train(makeDataset(128), 2);
      assertThat(eq128_2.magByteSize()).isEqualTo(32);
      assertThat(eq128_2.encodedByteSize()).isEqualTo(48);
    }

    @Test
    void compressionRatio_accountsForOverhead() {
      // dim=128, 4-bit: original=512, compressed=80 + 6*4=104 → ratio=512/104≈4.92
      var eq = ExtendedRaBitQuantizer.train(makeDataset(128), 4);
      int compressedBytes =
          eq.encodedByteSize() + ExtendedRaBitQuantizer.NUM_CORRECTIONS * Float.BYTES;
      float expected = (128 * 4.0f) / compressedBytes;
      assertThat(eq.compressionRatio()).isCloseTo(expected, within(0.01f));
    }

    @Test
    void magnitudeCodes_inValidRange() {
      float[][] vectors = generateNormalizedVectors(50, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      int bits = 4;
      var eq = ExtendedRaBitQuantizer.train(dataset, bits);
      var compressed = eq.encodeAll(dataset);
      int maxLevel = (1 << bits) - 1;

      for (int i = 0; i < 10; i++) {
        byte[] magCodes = compressed.getMagCodes(i);
        // Unpack and verify all indices are in [0, maxLevel]
        int[] indices =
            ExtendedRaBitQuantizer.unpackMagnitudes(magCodes, eq.paddedDimension(), bits);
        for (int d = 0; d < eq.paddedDimension(); d++) {
          assertThat(indices[d])
              .as("Magnitude index at dim %d for vector %d", d, i)
              .isBetween(0, maxLevel);
        }
      }
    }
  }

  // --- Greedy Quantize Tests ---

  @Nested
  @Tag("unit")
  class GreedyQuantizeTests {

    @Test
    void greedyQuantize_maximizesCosine() {
      Random rng = new Random(42L);
      int dim = 128;
      float[] magnitudes = new float[dim];
      for (int d = 0; d < dim; d++) {
        magnitudes[d] = Math.abs((float) rng.nextGaussian());
      }

      int[] indices = ExtendedRaBitQuantizer.greedyQuantize(magnitudes, 4);
      int maxLevel = 15;

      // Verify all indices in range
      for (int idx : indices) {
        assertThat(idx).isBetween(0, maxLevel);
      }

      // Compute cosine similarity between (indices + 0.5) and magnitudes
      // Greedy quantize should produce a good approximation
      float dot = 0f, normA = 0f, normB = 0f;
      for (int d = 0; d < dim; d++) {
        float q = indices[d] + 0.5f;
        dot += q * magnitudes[d];
        normA += q * q;
        normB += magnitudes[d] * magnitudes[d];
      }
      float cosine = dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
      assertThat(cosine).as("Greedy quantize cosine similarity").isGreaterThan(0.9f);
    }

    @Test
    void greedyQuantize_differentBits_differentResolution() {
      Random rng = new Random(42L);
      int dim = 64;
      float[] magnitudes = new float[dim];
      for (int d = 0; d < dim; d++) {
        magnitudes[d] = Math.abs((float) rng.nextGaussian());
      }

      // More bits should give better approximation
      float prevCosine = 0f;
      for (int bits = 2; bits <= 8; bits++) {
        int[] indices = ExtendedRaBitQuantizer.greedyQuantize(magnitudes, bits);
        float dot = 0f, normA = 0f, normB = 0f;
        for (int d = 0; d < dim; d++) {
          float q = indices[d] + 0.5f;
          dot += q * magnitudes[d];
          normA += q * q;
          normB += magnitudes[d] * magnitudes[d];
        }
        float cosine = dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
        assertThat(cosine)
            .as("Cosine for %d bits should be >= cosine for %d bits", bits, bits - 1)
            .isGreaterThanOrEqualTo(prevCosine - 0.01f); // small tolerance for numerical noise
        prevCosine = cosine;
      }
    }
  }

  // --- Correction Factor Tests ---

  @Nested
  @Tag("unit")
  class CorrectionFactors {

    @Test
    void corrections_sqrXMatchesSquaredNorm() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      float[] centroid = eq.centroid();

      var compressed = eq.encodeAll(dataset);
      for (int i = 0; i < 10; i++) {
        float[] corrections = compressed.getCorrections(i);
        float expectedSqrX = 0f;
        for (int d = 0; d < 128; d++) {
          float v = vectors[i][d] - centroid[d];
          expectedSqrX += v * v;
        }
        assertThat(corrections[ExtendedRaBitQuantizer.IDX_SQR_X])
            .isCloseTo(expectedSqrX, within(expectedSqrX * 1e-3f + 1e-4f));
      }
    }

    @Test
    void corrections_x0InExpectedRange() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      for (int i = 0; i < 200; i++) {
        float x0 = compressed.getCorrections(i)[ExtendedRaBitQuantizer.IDX_X0];
        // x0 = sum|rotated[d]| / sqrt(D) — always in (0, 1] by Cauchy-Schwarz for non-zero unit
        // vectors
        assertThat(x0).as("x0 for vector %d", i).isGreaterThan(0f).isLessThanOrEqualTo(1f);
      }
    }

    @Test
    void corrections_matchDeclaredSize() {
      float[][] vectors = generateNormalizedVectors(10, 64, 7L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      float[] corr = compressed.getCorrections(0);
      assertThat(corr).hasSize(ExtendedRaBitQuantizer.NUM_CORRECTIONS);
      for (float v : corr) {
        assertThat(v).isFinite();
      }
    }

    @Test
    void corrections_sqrXAndX0ConsistentWithOneBit() {
      // sqrX and x0 should match 1-bit RaBitQ (same encoding pipeline for sign bits).
      // factorPpc and factorIp are intentionally different: multi-bit uses x0_multi and norm_code
      // instead of x0_1bit and sqrt(D), giving tighter estimates.
      float[][] vectors = generateNormalizedVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var rq = RaBitQuantizer.train(dataset);
      var rqCompressed = rq.encodeAll(dataset);

      // Use same rotation for comparison
      var eq = ExtendedRaBitQuantizer.train(dataset, 4, rq.rotation());
      var eqCompressed = eq.encodeAll(dataset);

      for (int i = 0; i < 20; i++) {
        float[] rqCorr = rqCompressed.getCorrections(i);
        float[] eqCorr = eqCompressed.getCorrections(i);

        // sqrX should be identical (same centering pipeline)
        assertThat(eqCorr[ExtendedRaBitQuantizer.IDX_SQR_X])
            .isCloseTo(rqCorr[RaBitQuantizer.IDX_SQR_X], within(1e-3f));
        // x0 (1-bit quality) should be identical (same sign-bit encoding)
        assertThat(eqCorr[ExtendedRaBitQuantizer.IDX_X0])
            .isCloseTo(rqCorr[RaBitQuantizer.IDX_X0], within(1e-3f));

        // factorIp_multi should have the same sign but larger magnitude (tighter estimate)
        // because x0_multi >= x0_1bit (magnitude codes add precision)
        float rqFactorIp = rqCorr[RaBitQuantizer.IDX_FACTOR_IP];
        float eqFactorIp = eqCorr[ExtendedRaBitQuantizer.IDX_FACTOR_IP];
        assertThat(eqFactorIp).isNotEqualTo(0f);
        assertThat(eqFactorIp)
            .as("Extended RaBitQ factorIp should differ from 1-bit RaBitQ")
            .isNotCloseTo(rqFactorIp, within(1e-6f));

        float rqFactorPpc = rqCorr[RaBitQuantizer.IDX_FACTOR_PPC];
        float eqFactorPpc = eqCorr[ExtendedRaBitQuantizer.IDX_FACTOR_PPC];
        assertThat(eqFactorPpc)
            .as("Extended RaBitQ popcount correction should differ from 1-bit RaBitQ")
            .isNotCloseTo(rqFactorPpc, within(1e-6f));
      }
    }
  }

  // --- Scoring Tests ---

  @Nested
  @Tag("unit")
  class ScoringTests {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void scoreCorrelatesWithTrueDistance(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction sf = compressed.scoreFunctionFor(query, simFunc);

      float[] approxScores = new float[vectors.length];
      float[] trueScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        approxScores[i] = sf.score(i);
        trueScores[i] = simFunc.compare(query, vectors[i]);
      }

      int[] approxTop10 = topK(approxScores, 10);
      int[] trueTop10 = topK(trueScores, 10);
      int overlap = countOverlap(approxTop10, trueTop10);
      assertThat(overlap)
          .as("Extended RaBitQ %s: top-10 overlap", simFunc)
          .isGreaterThanOrEqualTo(4);
    }

    @Test
    void extendedRaBitQ_betterThan1bitRaBitQ() {
      float[][] vectors = generateNormalizedVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      var rq = RaBitQuantizer.train(dataset);
      var rqCompressed = rq.encodeAll(dataset);

      var eq = ExtendedRaBitQuantizer.train(dataset, 4, rq.rotation());
      var eqCompressed = eq.encodeAll(dataset);

      int rqTotal = 0;
      int eqTotal = 0;
      int numQueries = 10;
      for (int qi = 0; qi < numQueries; qi++) {
        float[] query = vectors[qi];
        ScoreFunction sfRq = rqCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
        ScoreFunction sfEq = eqCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

        float[] trueScores = new float[vectors.length];
        float[] rqScores = new float[vectors.length];
        float[] eqScores = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
          trueScores[i] = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
          rqScores[i] = sfRq.score(i);
          eqScores[i] = sfEq.score(i);
        }

        int[] trueTop20 = topK(trueScores, 20);
        rqTotal += countOverlap(trueTop20, topK(rqScores, 20));
        eqTotal += countOverlap(trueTop20, topK(eqScores, 20));
      }

      // 4-bit stores 4x more bits/dim than 1-bit; >= allows a regression where multi-bit
      // degrades to 1-bit quality. Use strict > to catch that.
      assertThat(eqTotal)
          .as(
              "Extended RaBitQ (4-bit) must strictly outperform 1-bit RaBitQ "
                  + "(rqTotal=%d) — 4x more bits/dim implies better distance estimates",
              rqTotal)
          .isGreaterThan(rqTotal);
    }

    @Test
    void accuracy_improvesWithMoreBits() {
      float[][] vectors = generateNormalizedVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      int prevOverlap = 0;
      for (int bits : new int[] {2, 4, 8}) {
        var eq = ExtendedRaBitQuantizer.train(dataset, bits);
        var compressed = eq.encodeAll(dataset);

        int totalOverlap = 0;
        int numQueries = 5;
        for (int qi = 0; qi < numQueries; qi++) {
          float[] query = vectors[qi];
          ScoreFunction sf = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

          float[] trueScores = new float[vectors.length];
          float[] approxScores = new float[vectors.length];
          for (int i = 0; i < vectors.length; i++) {
            trueScores[i] = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
            approxScores[i] = sf.score(i);
          }

          int[] trueTop20 = topK(trueScores, 20);
          totalOverlap += countOverlap(trueTop20, topK(approxScores, 20));
        }

        assertThat(totalOverlap)
            .as("Extended RaBitQ %d-bit overlap should be >= %d-bit", bits, bits)
            .isGreaterThanOrEqualTo(prevOverlap);
        prevOverlap = totalOverlap;
      }
    }

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void coarseScoreFunction_correlatesWithTrueDistance(SimilarityFunction simFunc) {
      // Layer 1 (sign-only) should still rank-correlate, just less accurately than Layer 2
      float[][] vectors = generateNormalizedVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      float[] query = vectors[0];
      ScoreFunction coarseSf = compressed.coarseScoreFunctionFor(query, simFunc);

      float[] coarseScores = new float[vectors.length];
      float[] trueScores = new float[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        coarseScores[i] = coarseSf.score(i);
        trueScores[i] = simFunc.compare(query, vectors[i]);
      }

      int[] coarseTop10 = topK(coarseScores, 10);
      int[] trueTop10 = topK(trueScores, 10);
      int overlap = countOverlap(coarseTop10, trueTop10);
      // Layer 1 is coarse — expect at least some correlation (>= 2 of top-10)
      assertThat(overlap)
          .as("Layer 1 coarse %s: top-10 overlap", simFunc)
          .isGreaterThanOrEqualTo(2);
    }

    @Test
    void layer2_betterThanLayer1() {
      // Multi-bit (Layer 2) should be more accurate than sign-only (Layer 1)
      float[][] vectors = generateNormalizedVectors(1000, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      int layer1Total = 0;
      int layer2Total = 0;
      int numQueries = 10;
      for (int qi = 0; qi < numQueries; qi++) {
        float[] query = vectors[qi];
        ScoreFunction sfLayer1 =
            compressed.coarseScoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
        ScoreFunction sfLayer2 = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

        float[] trueScores = new float[vectors.length];
        float[] l1Scores = new float[vectors.length];
        float[] l2Scores = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
          trueScores[i] = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
          l1Scores[i] = sfLayer1.score(i);
          l2Scores[i] = sfLayer2.score(i);
        }

        int[] trueTop20 = topK(trueScores, 20);
        layer1Total += countOverlap(trueTop20, topK(l1Scores, 20));
        layer2Total += countOverlap(trueTop20, topK(l2Scores, 20));
      }

      assertThat(layer2Total)
          .as("Layer 2 (multi-bit) should have >= top-20 overlap than Layer 1 (sign-only)")
          .isGreaterThanOrEqualTo(layer1Total);
    }

    @Test
    void rescoring_withQueryNormReducesErrorOnScaledQueries() {
      // Rescoring folds queryNorm into factorIp/factorPpc; it should outperform the legacy
      // stored-factor estimator when queries have larger norms.
      float[][] vectors = generateNormalizedVectors(300, 64, 11L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      float[] query = vectors[0];
      float[] scaledQuery = Arrays.copyOf(query, query.length);
      for (int i = 0; i < scaledQuery.length; i++) {
        scaledQuery[i] *= 2.5f;
      }

      PreparedQueryMetrics prep = prepareQueryForTest(scaledQuery, eq);

      int improved = 0;
      int total = 120;
      for (int i = 0; i < total; i++) {
        float trueDist = squaredL2(scaledQuery, vectors[i]);
        float newDist =
            ExtendedRaBitQuantizedVectors.estimateL2Multi(
                compressed.getSignCodes(i),
                compressed.getMagCodes(i),
                compressed.getCorrections(i),
                prep.queryBytes(),
                prep.vl(),
                prep.widthOver255(),
                prep.sumQ(),
                prep.sqrY(),
                prep.queryNorm(),
                4);
        float oldDist =
            estimateL2MultiWithoutRescore(
                compressed.getSignCodes(i),
                compressed.getMagCodes(i),
                compressed.getCorrections(i),
                prep.queryBytes(),
                prep.vl(),
                prep.widthOver255(),
                prep.sumQ(),
                prep.sqrY(),
                4);
        if (Math.abs(newDist - trueDist) <= Math.abs(oldDist - trueDist)) {
          improved++;
        }
      }

      assertThat(improved)
          .as("Rescoring should reduce or match error for most candidates")
          .isGreaterThanOrEqualTo(total / 2);
    }

    @Test
    void totalBitCount_matchesNaiveCount() {
      Random rng = new Random(42L);
      int numLongs = 4;
      long[] bits = new long[numLongs];
      int expected = 0;
      for (int i = 0; i < numLongs; i++) {
        bits[i] = rng.nextLong();
        expected += Long.bitCount(bits[i]);
      }
      assertThat(ExtendedRaBitQuantizedVectors.totalBitCount(bits)).isEqualTo(expected);
    }
  }

  // --- Bit Packing Tests ---

  @Nested
  @Tag("unit")
  class BitPacking {

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4, 5, 6, 7, 8})
    void packMagnitudes_roundTrips(int bits) {
      Random rng = new Random(42L);
      int dim = 128;
      int maxLevel = (1 << bits) - 1;
      int[] indices = new int[dim];
      for (int d = 0; d < dim; d++) {
        indices[d] = rng.nextInt(maxLevel + 1);
      }

      byte[] packed = ExtendedRaBitQuantizer.packMagnitudes(indices, bits);
      int[] unpacked = ExtendedRaBitQuantizer.unpackMagnitudes(packed, dim, bits);

      assertThat(unpacked).isEqualTo(indices);
    }

    @Test
    void packMagnitudes_4bit_roundTrips() {
      // Special case: 4-bit packing (2 per byte)
      int dim = 64;
      int[] indices = new int[dim];
      for (int d = 0; d < dim; d++) {
        indices[d] = d % 16;
      }

      byte[] packed = ExtendedRaBitQuantizer.packMagnitudes(indices, 4);
      assertThat(packed).hasSize(32); // 64 * 4 / 8 = 32
      int[] unpacked = ExtendedRaBitQuantizer.unpackMagnitudes(packed, dim, 4);
      assertThat(unpacked).isEqualTo(indices);
    }

    @Test
    void signAndMagCodes_areIndependent() {
      // Verify that sign bits and magnitude codes don't interfere
      float[][] vectors = generateNormalizedVectors(50, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      for (int i = 0; i < 10; i++) {
        long[] signCodes = compressed.getSignCodes(i);
        byte[] magCodes = compressed.getMagCodes(i);

        // Sign codes should have values (not all zeros)
        boolean hasSignBits = false;
        for (long l : signCodes) {
          if (l != 0L) {
            hasSignBits = true;
            break;
          }
        }
        assertThat(hasSignBits).as("Sign codes for vector %d should have set bits", i).isTrue();

        // Magnitude codes should have values
        boolean hasMagValues = false;
        for (byte b : magCodes) {
          if (b != 0) {
            hasMagValues = true;
            break;
          }
        }
        assertThat(hasMagValues)
            .as("Magnitude codes for vector %d should have non-zero values", i)
            .isTrue();
      }
    }
  }

  // --- Default Rotation ---

  @Nested
  @Tag("unit")
  class DefaultRotation {

    @Test
    void train_withSeed_defaultsToGivensRotation() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      ExtendedRaBitQuantizer eq = ExtendedRaBitQuantizer.train(dataset, 4, 42L);
      assertThat(eq.rotation()).isInstanceOf(GivensRotation.class);
    }

    @Test
    void train_noSeed_defaultsToGivensRotation() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      ExtendedRaBitQuantizer eq = ExtendedRaBitQuantizer.train(dataset, 4);
      assertThat(eq.rotation()).isInstanceOf(GivensRotation.class);
    }
  }

  // --- Rotation Strategies ---

  @Nested
  @Tag("unit")
  class RotationStrategies {

    @Test
    void worksWithGivensRotation() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      int paddedDim = 128;
      Rotation givens = GivensRotation.generate(paddedDim, 42L);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4, givens);
      var compressed = eq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      float score = sf.score(1);
      assertThat(score).isGreaterThan(0f).isFinite();
    }

    @Test
    void worksWithQuaternionRotation() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      int paddedDim = 128;
      Rotation quaternion = QuaternionRotation.generate(paddedDim, 42L);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4, quaternion);
      var compressed = eq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      float score = sf.score(1);
      assertThat(score).isGreaterThan(0f).isFinite();
    }

    @Test
    void worksWithRandomRotation() {
      float[][] vectors = generateNormalizedVectors(200, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      int paddedDim = 128;
      Rotation random = RandomRotation.generate(paddedDim, 42L);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4, random);
      var compressed = eq.encodeAll(dataset);

      ScoreFunction sf = compressed.scoreFunctionFor(vectors[0], SimilarityFunction.EUCLIDEAN);
      float score = sf.score(1);
      assertThat(score).isGreaterThan(0f).isFinite();
    }
  }

  // --- SIFT Small Dataset Tests ---

  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.ExtendedRaBitQuantizerTest#siftAvailable")
  class SiftSmallExtRaBitQ {

    @Test
    void recall_euclidean_4bit() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var eq = ExtendedRaBitQuantizer.train(dataset, 4);
      var compressed = eq.encodeAll(dataset);

      double recall = computeAverageRecall(compressed, base, queries, groundTruth, 10);
      assertThat(recall)
          .as("Extended RaBitQ 4-bit average recall@10 on SIFT Small")
          .isGreaterThan(0.50);
    }

    @Test
    void recall_betterThan1bitRaBitQ() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);

      var rq = RaBitQuantizer.train(dataset);
      var rqCompressed = rq.encodeAll(dataset);
      double rqRecall = computeAverageRecall(rqCompressed, base, queries, groundTruth, 10);

      var eq = ExtendedRaBitQuantizer.train(dataset, 4, rq.rotation());
      var eqCompressed = eq.encodeAll(dataset);
      double eqRecall = computeAverageRecall(eqCompressed, base, queries, groundTruth, 10);

      assertThat(eqRecall)
          .as("Extended RaBitQ 4-bit recall should be >= 1-bit RaBitQ recall")
          .isGreaterThanOrEqualTo(rqRecall);
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

  // --- Helper methods ---

  private static PreparedQueryMetrics prepareQueryForTest(
      float[] query, ExtendedRaBitQuantizer quantizer) {
    int paddedDim = quantizer.paddedDimension();
    float[] centeredQuery = new float[paddedDim];
    System.arraycopy(query, 0, centeredQuery, 0, query.length);
    VectorUtil.subInPlace(centeredQuery, quantizer.paddedCentroid());

    float sqrY = VectorUtil.dotProduct(centeredQuery, centeredQuery);
    float queryNorm = (float) Math.sqrt(sqrY);

    float[] queryRotated = quantizer.rotation().rotate(centeredQuery);

    float vl = Float.POSITIVE_INFINITY;
    float vh = Float.NEGATIVE_INFINITY;
    for (float v : queryRotated) {
      if (v < vl) vl = v;
      if (v > vh) vh = v;
    }
    float width = vh - vl;
    float scale = width > 0 ? 255.0f / width : 0f;

    byte[] queryBytes = new byte[paddedDim];
    int sumQ = 0;
    for (int d = 0; d < paddedDim; d++) {
      int q = Math.round((queryRotated[d] - vl) * scale);
      q = Math.max(0, Math.min(255, q));
      queryBytes[d] = (byte) q;
      sumQ += q;
    }

    return new PreparedQueryMetrics(queryBytes, vl, width / 255.0f, sumQ, sqrY, queryNorm);
  }

  private static float estimateL2MultiWithoutRescore(
      long[] storedSignBits,
      byte[] storedMagCodes,
      float[] corrections,
      byte[] queryBytes,
      float vl,
      float widthOver255,
      int sumQ,
      float sqrY,
      int bits) {
    float sqrX = corrections[ExtendedRaBitQuantizer.IDX_SQR_X];
    float factorPpc = corrections[ExtendedRaBitQuantizer.IDX_FACTOR_PPC];
    float factorIp = corrections[ExtendedRaBitQuantizer.IDX_FACTOR_IP];

    int signedMagIp =
        ExtendedRaBitQuantizedVectors.asymmetricByteTimesSignedMag(
            queryBytes, storedMagCodes, storedSignBits, bits);
    int ip1bit = RaBitQuantizedVectors.asymmetricByteTimesBit(queryBytes, storedSignBits);
    float bFull = signedMagIp + 0.5f * (2 * ip1bit - sumQ);
    return sqrX + sqrY + factorPpc * vl + factorIp * widthOver255 * bFull;
  }

  private static float squaredL2(float[] a, float[] b) {
    float acc = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      acc += d * d;
    }
    return acc;
  }

  private static ArrayVectorDataset makeDataset(int dim) {
    float[][] vectors = new float[10][dim];
    Random rng = new Random(42L);
    for (int i = 0; i < 10; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = rng.nextFloat() * 2 - 1;
      }
    }
    return new ArrayVectorDataset(vectors);
  }

  private static float[][] generateVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] vectors = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        vectors[i][d] = rng.nextFloat() * 2 - 1;
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

  private record PreparedQueryMetrics(
      byte[] queryBytes, float vl, float widthOver255, int sumQ, float sqrY, float queryNorm) {}
}
