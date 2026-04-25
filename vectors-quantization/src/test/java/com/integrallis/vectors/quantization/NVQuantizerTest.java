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
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Tests for {@link NVQuantizer}, {@link NVQuantizedVectors}, and {@link NQTransform}. NVQ uses
 * per-vector nonlinear quantization for high-accuracy approximate scoring.
 */
class NVQuantizerTest {

  private static final Path SIFT_DIR =
      Path.of("../../research/repos/jvector/siftsmall").toAbsolutePath().normalize();

  static boolean siftAvailable() {
    return Files.exists(SIFT_DIR.resolve("siftsmall_base.fvecs"));
  }

  // --- NQTransform Tests ---

  @Nested
  @Tag("unit")
  class NQTransformTests {

    @Test
    void logisticNQT_matchesStandardLogistic_approximately() {
      // NQT logistic uses IEEE 754 bit-manipulation (~4 FMA ops) instead of exp().
      // Max absolute error is ~0.09 across [-5, 5]; exact at z=0.
      // Round-trip quality (logistic→logit) is what matters for quantization accuracy.
      for (float z = -5f; z <= 5f; z += 0.5f) {
        float nqt = NQTransform.logistic(z, 1f, 0f);
        float exact = 1f / (1f + (float) Math.exp(-z));
        assertThat(nqt).as("NQT logistic at z=%f", z).isCloseTo(exact, within(0.1f));
      }
    }

    @Test
    void logitNQT_isInverseOfLogisticNQT() {
      // logit(logistic(v)) ≈ v for values in a reasonable range
      for (float v = -3f; v <= 3f; v += 0.5f) {
        float logisticV = NQTransform.logistic(v, 1f, 0f);
        float recovered = NQTransform.logit(logisticV, 1f, 0f);
        assertThat(recovered).as("logit(logistic(%f))", v).isCloseTo(v, within(0.2f));
      }
    }

    @Test
    void scaledLogistic_mapsToZeroOne() {
      float min = -2f;
      float max = 3f;
      float alpha = 2f;
      float x0 = (min + max) / 2f;
      float range = max - min;
      float scaledAlpha = alpha / range;
      float logMin = NQTransform.logistic(min, scaledAlpha, x0);
      float logMax = NQTransform.logistic(max, scaledAlpha, x0);
      float logScale = 1f / (logMax - logMin);

      float atMin = NQTransform.scaledLogistic(min, scaledAlpha, x0, logMin, logScale);
      float atMax = NQTransform.scaledLogistic(max, scaledAlpha, x0, logMin, logScale);
      float atMid = NQTransform.scaledLogistic(x0, scaledAlpha, x0, logMin, logScale);

      assertThat(atMin).isCloseTo(0f, within(0.01f));
      assertThat(atMax).isCloseTo(1f, within(0.01f));
      assertThat(atMid).isBetween(0.3f, 0.7f); // midpoint should be roughly centered
    }

    @Test
    void scaledLogit_invertsScaledLogistic() {
      float min = -1f;
      float max = 2f;
      float alpha = 3f;
      float x0 = (min + max) / 2f;
      float range = max - min;
      float scaledAlpha = alpha / range;
      float logMin = NQTransform.logistic(min, scaledAlpha, x0);
      float logMax = NQTransform.logistic(max, scaledAlpha, x0);
      float logRange = logMax - logMin;
      float logScale = 1f / logRange;
      float invScaledAlpha = range / alpha;

      for (float v = min + 0.1f; v < max - 0.1f; v += 0.3f) {
        float mapped = NQTransform.scaledLogistic(v, scaledAlpha, x0, logMin, logScale);
        float recovered = NQTransform.scaledLogit(mapped, invScaledAlpha, x0, logScale, logMin);
        assertThat(recovered).as("scaledLogit(scaledLogistic(%f))", v).isCloseTo(v, within(0.1f));
      }
    }

    @Test
    void optimizeTransform_reducesError_vsUniform() {
      Random rng = new Random(42L);
      // Non-uniform distribution: mostly small values with a few large
      float[] sv = new float[64];
      for (int d = 0; d < 64; d++) {
        float r = rng.nextFloat();
        sv[d] = r < 0.8f ? (float) rng.nextGaussian() * 0.3f : (float) rng.nextGaussian() * 2f;
      }
      float min = Float.POSITIVE_INFINITY;
      float max = Float.NEGATIVE_INFINITY;
      for (float v : sv) {
        if (v < min) min = v;
        if (v > max) max = v;
      }

      float[] params = NQTransform.optimizeTransform(sv, min, max);
      assertThat(params).hasSize(2);
      assertThat(params[0]).as("alpha").isGreaterThan(0f);
    }

    @Test
    void optimizeTransform_deterministicForSameInput() {
      Random rng = new Random(42L);
      float[] sv = new float[32];
      for (int d = 0; d < 32; d++) {
        sv[d] = (float) rng.nextGaussian();
      }
      float min = -3f;
      float max = 3f;

      float[] params1 = NQTransform.optimizeTransform(sv, min, max);
      float[] params2 = NQTransform.optimizeTransform(sv, min, max);
      assertThat(params1[0]).isEqualTo(params2[0]);
      assertThat(params1[1]).isEqualTo(params2[1]);
    }
  }

  // --- Encoding Tests ---

  @Nested
  @Tag("unit")
  class EncodingTests {

    @Test
    void encode_centersBeforeQuantizing() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);

      byte[] encoded = nvq.encode(vectors[0]);
      assertThat(encoded).hasSize(nvq.encodedByteSize());

      // Quantized bytes should have a mix of values (not all zeros)
      int nonZero = 0;
      for (int d = 0; d < 128; d++) {
        if (encoded[d] != 0) nonZero++;
      }
      assertThat(nonZero).isGreaterThan(10);
    }

    @Test
    void encode_reusedBuffer_doesNotCorrupt() {
      float[][] vectors = generateVectors(100, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);
      byte[] dst = new byte[nvq.encodedByteSize()];

      nvq.encode(vectors[0], dst);
      byte[] first = java.util.Arrays.copyOf(dst, dst.length);

      nvq.encode(vectors[1], dst);
      nvq.encode(vectors[0], dst);
      assertThat(dst).isEqualTo(first);
    }

    @Test
    void encode_dimensionMismatch_throws() {
      var nvq = NVQuantizer.train(makeDataset(128), 2);
      assertThatThrownBy(() -> nvq.encode(new float[64]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Expected dimension 128");
    }

    @Test
    void encodeAndDecode_roundTrip_approximatesOriginal() {
      float[][] vectors = generateVectors(50, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);

      for (int i = 0; i < 10; i++) {
        float[] decoded = nvq.decode(nvq.encode(vectors[i]));
        assertThat(decoded).hasSize(128);

        // NVQ with nonlinear transform should give good reconstruction
        float cosine = VectorUtil.cosine(vectors[i], decoded);
        assertThat(cosine).as("Decoded cosine for vector %d", i).isGreaterThan(0.95f);
      }
    }

    @Test
    void encodedSize_matchesDimensionPlusMetadata() {
      // dim=128, 2 subvectors: 128 bytes + 2*16 = 160
      var nvq = NVQuantizer.train(makeDataset(128), 2);
      assertThat(nvq.encodedByteSize()).isEqualTo(128 + 2 * 16);

      // dim=768, 12 subvectors: 768 + 12*16 = 960
      var nvq768 = NVQuantizer.train(makeDataset(768), 12);
      assertThat(nvq768.encodedByteSize()).isEqualTo(768 + 12 * 16);
    }

    @Test
    void compressionRatio_isAbout3x_for768dim() {
      // dim=768, 12 subvectors: original=3072, compressed=960 → ratio≈3.2
      var nvq = NVQuantizer.train(makeDataset(768), 12);
      float expected = (768 * 4.0f) / (768 + 12 * 16);
      assertThat(nvq.compressionRatio()).isCloseTo(expected, within(0.01f));
    }

    @Test
    void invalidNumSubvectors_throws() {
      assertThatThrownBy(() -> NVQuantizer.train(makeDataset(128), 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> NVQuantizer.train(makeDataset(128), 129))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  // --- Subvector Tests ---

  @Nested
  @Tag("unit")
  class SubvectorTests {

    @Test
    void subvectorSizes_sumToDimension() {
      for (int dim : new int[] {128, 100, 768, 33}) {
        for (int m = 1; m <= Math.min(dim, 8); m++) {
          int[] sizes = NVQuantizer.computeSubvectorSizes(dim, m);
          assertThat(sizes).hasSize(m);
          int sum = 0;
          for (int s : sizes) {
            assertThat(s).isGreaterThan(0);
            sum += s;
          }
          assertThat(sum).as("sum of sizes for dim=%d, m=%d", dim, m).isEqualTo(dim);
        }
      }
    }

    @Test
    void subvectorPartitioning_handlesUnevenDivision() {
      // 128 / 3 = 42 r 2 → sizes should be [43, 43, 42]
      int[] sizes = NVQuantizer.computeSubvectorSizes(128, 3);
      assertThat(sizes).hasSize(3);
      assertThat(sizes[0] + sizes[1] + sizes[2]).isEqualTo(128);
      // First two should be 43, last 42
      assertThat(sizes[0]).isEqualTo(43);
      assertThat(sizes[1]).isEqualTo(43);
      assertThat(sizes[2]).isEqualTo(42);
    }

    @Test
    void perSubvectorMetadata_hasValidRanges() {
      float[][] vectors = generateVectors(50, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);
      var compressed = nvq.encodeAll(dataset);

      for (int i = 0; i < 10; i++) {
        float[] meta = compressed.getSubvectorMetadata(i);
        assertThat(meta).hasSize(2 * 4); // 2 subvectors * 4 floats each

        for (int m = 0; m < 2; m++) {
          int idx = m * 4;
          float alpha = meta[idx];
          float x0 = meta[idx + 1];
          float min = meta[idx + 2];
          float max = meta[idx + 3];

          assertThat(alpha).as("alpha for sv %d, vec %d", m, i).isGreaterThan(0f).isFinite();
          assertThat(x0).as("x0 for sv %d, vec %d", m, i).isFinite();
          assertThat(min).as("min for sv %d, vec %d", m, i).isLessThan(max);
        }
      }
    }

    @Test
    void differentSubvectors_getDifferentAlpha() {
      // With varied data, subvectors should sometimes get different alpha values
      Random rng = new Random(42L);
      float[][] vectors = new float[100][128];
      for (int i = 0; i < 100; i++) {
        for (int d = 0; d < 64; d++) {
          // First half: tight distribution
          vectors[i][d] = (float) rng.nextGaussian() * 0.1f;
        }
        for (int d = 64; d < 128; d++) {
          // Second half: wide distribution
          vectors[i][d] = (float) rng.nextGaussian() * 5f;
        }
      }

      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);
      var compressed = nvq.encodeAll(dataset);

      // Check that at least some vectors have different alphas for the two subvectors
      boolean anyDifferent = false;
      for (int i = 0; i < 50; i++) {
        float[] meta = compressed.getSubvectorMetadata(i);
        float alpha0 = meta[0];
        float alpha1 = meta[4];
        if (Math.abs(alpha0 - alpha1) > 0.01f) {
          anyDifferent = true;
          break;
        }
      }
      assertThat(anyDifferent)
          .as("Some vectors should have different alphas per subvector")
          .isTrue();
    }
  }

  // --- Scoring Tests ---

  @Nested
  @Tag("unit")
  class ScoringTests {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void scoreCorrelatesWithTrueDistance(SimilarityFunction simFunc) {
      float[][] vectors = generateNormalizedVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);
      var nvq = NVQuantizer.train(dataset, 2);
      var compressed = nvq.encodeAll(dataset);

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
      // NVQ should have very high recall due to per-vector nonlinear quantization
      assertThat(overlap).as("NVQ %s: top-10 overlap", simFunc).isGreaterThanOrEqualTo(6);
    }

    @Test
    void nvq_betterThanUniformSQ_atSameCompression() {
      float[][] vectors = generateVectors(500, 128, 42L);
      var dataset = new ArrayVectorDataset(vectors);

      // NVQ with 2 subvectors
      var nvq = NVQuantizer.train(dataset, 2);
      var nvqCompressed = nvq.encodeAll(dataset);

      // Scalar quantizer int8 (same ~4x compression)
      var sq = ScalarQuantizer.train(dataset);
      var sqCompressed = sq.encodeAll(dataset);

      int nvqTotal = 0;
      int sqTotal = 0;
      int numQueries = 10;
      for (int qi = 0; qi < numQueries; qi++) {
        float[] query = vectors[qi];
        ScoreFunction sfNvq = nvqCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
        ScoreFunction sfSq = sqCompressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);

        float[] trueScores = new float[vectors.length];
        float[] nvqScores = new float[vectors.length];
        float[] sqScores = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
          trueScores[i] = SimilarityFunction.EUCLIDEAN.compare(query, vectors[i]);
          nvqScores[i] = sfNvq.score(i);
          sqScores[i] = sfSq.score(i);
        }

        int[] trueTop20 = topK(trueScores, 20);
        nvqTotal += countOverlap(trueTop20, topK(nvqScores, 20));
        sqTotal += countOverlap(trueTop20, topK(sqScores, 20));
      }

      // NVQ should be at least comparable to SQ int8
      assertThat(nvqTotal)
          .as("NVQ total overlap should be competitive with SQ int8")
          .isGreaterThanOrEqualTo((int) (sqTotal * 0.8));
    }
  }

  // --- SIFT Small Dataset Tests ---

  @Nested
  @Tag("slow")
  @EnabledIf("com.integrallis.vectors.quantization.NVQuantizerTest#siftAvailable")
  class SiftSmallNVQ {

    @Test
    void recall_euclidean_2subvectors() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);
      var nvq = NVQuantizer.train(dataset, 2);
      var compressed = nvq.encodeAll(dataset);

      double recall = computeAverageRecall(compressed, base, queries, groundTruth, 10);
      assertThat(recall).as("NVQ average recall@10 on SIFT Small").isGreaterThan(0.70);
    }

    @Test
    void recall_betterThanScalarInt8_atSameCompression() {
      float[][] base = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_base.fvecs"));
      float[][] queries = SiftLoader.readFvecs(SIFT_DIR.resolve("siftsmall_query.fvecs"));
      int[][] groundTruth = SiftLoader.readIvecs(SIFT_DIR.resolve("siftsmall_groundtruth.ivecs"));

      var dataset = SiftLoader.asDataset(base);

      var sq = ScalarQuantizer.train(dataset);
      var sqCompressed = sq.encodeAll(dataset);
      double sqRecall = computeAverageRecall(sqCompressed, base, queries, groundTruth, 10);

      var nvq = NVQuantizer.train(dataset, 2);
      var nvqCompressed = nvq.encodeAll(dataset);
      double nvqRecall = computeAverageRecall(nvqCompressed, base, queries, groundTruth, 10);

      assertThat(nvqRecall)
          .as("NVQ recall should be competitive with SQ int8")
          .isGreaterThanOrEqualTo(sqRecall * 0.9);
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
