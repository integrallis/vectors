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

import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link LloydMaxCodebook}. */
class LloydMaxCodebookTest {

  @Nested
  @Tag("unit")
  class BasicProperties {

    @Test
    void oneBit_centroidsAreSymmetric() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 1);
      assertThat(codebook.numLevels()).isEqualTo(2);
      assertThat(codebook.bits()).isEqualTo(1);
      float[] centroids = codebook.centroids();
      // Symmetric around 0
      assertThat(centroids[0]).isNegative();
      assertThat(centroids[1]).isPositive();
      assertThat(centroids[0]).isCloseTo(-centroids[1], within(1e-6f));
      // Boundary at 0
      float[] boundaries = codebook.boundaries();
      assertThat(boundaries).hasSize(1);
      assertThat(boundaries[0]).isCloseTo(0f, within(1e-6f));
    }

    @Test
    void twoBit_hasFourLevels() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 2);
      assertThat(codebook.numLevels()).isEqualTo(4);
      float[] centroids = codebook.centroids();
      assertThat(centroids).hasSize(4);
      float[] boundaries = codebook.boundaries();
      assertThat(boundaries).hasSize(3);
    }

    @Test
    void fourBit_has16Levels() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      assertThat(codebook.numLevels()).isEqualTo(16);
      assertThat(codebook.centroids()).hasSize(16);
      assertThat(codebook.boundaries()).hasSize(15);
    }

    @Test
    void eightBit_has256Levels() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 8);
      assertThat(codebook.numLevels()).isEqualTo(256);
      assertThat(codebook.centroids()).hasSize(256);
      assertThat(codebook.boundaries()).hasSize(255);
    }

    @Test
    void centroidsAreSorted() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      float[] centroids = codebook.centroids();
      for (int i = 1; i < centroids.length; i++) {
        assertThat(centroids[i]).isGreaterThan(centroids[i - 1]);
      }
    }

    @Test
    void boundariesAreSorted() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      float[] boundaries = codebook.boundaries();
      for (int i = 1; i < boundaries.length; i++) {
        assertThat(boundaries[i]).isGreaterThan(boundaries[i - 1]);
      }
    }

    @Test
    void boundariesAreBetweenCentroids() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      float[] centroids = codebook.centroids();
      float[] boundaries = codebook.boundaries();
      for (int i = 0; i < boundaries.length; i++) {
        assertThat(boundaries[i]).isGreaterThan(centroids[i]);
        assertThat(boundaries[i]).isLessThan(centroids[i + 1]);
      }
    }

    @Test
    void invalidBits_throws() {
      assertThatThrownBy(() -> LloydMaxCodebook.compute(128, 0))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> LloydMaxCodebook.compute(128, 9))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void invalidDimension_throws() {
      assertThatThrownBy(() -> LloydMaxCodebook.compute(0, 4))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  @Tag("unit")
  class QuantizeDequantize {

    @Test
    void quantize_returnsNearestCentroidIndex() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 2);
      float[] centroids = codebook.centroids();

      // Each centroid should quantize to itself
      for (int i = 0; i < centroids.length; i++) {
        assertThat(codebook.quantize(centroids[i])).isEqualTo(i);
      }
    }

    @Test
    void dequantize_returnsCentroidValue() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      float[] centroids = codebook.centroids();
      for (int i = 0; i < centroids.length; i++) {
        assertThat(codebook.dequantize(i)).isEqualTo(centroids[i]);
      }
    }

    @Test
    void quantizeDequantize_roundTrip_approximatesOriginal() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      float sigma = 1f / (float) Math.sqrt(128);
      Random rng = new Random(42L);

      double totalError = 0;
      int n = 1000;
      for (int i = 0; i < n; i++) {
        float value = (float) (rng.nextGaussian() * sigma);
        int idx = codebook.quantize(value);
        float reconstructed = codebook.dequantize(idx);
        totalError += (value - reconstructed) * (value - reconstructed);
      }
      double mse = totalError / n;
      // MSE should be small relative to variance
      double variance = sigma * sigma;
      assertThat(mse).isLessThan(variance); // quantization must reduce error
    }

    @Test
    void extremeValues_clampToEdgeCentroids() {
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4);
      // Very large positive value should map to last centroid
      int idxMax = codebook.quantize(100f);
      assertThat(idxMax).isEqualTo(codebook.numLevels() - 1);
      // Very large negative value should map to first centroid
      int idxMin = codebook.quantize(-100f);
      assertThat(idxMin).isEqualTo(0);
    }
  }

  @Nested
  @Tag("unit")
  class MSEOptimality {

    @Test
    void mseDecreases_withMoreBits() {
      int dim = 128;
      float sigma = 1f / (float) Math.sqrt(dim);
      Random rng = new Random(42L);

      // Generate test samples from N(0, 1/d)
      int n = 10000;
      float[] samples = new float[n];
      for (int i = 0; i < n; i++) {
        samples[i] = (float) (rng.nextGaussian() * sigma);
      }

      double prevMse = Double.MAX_VALUE;
      // MSE loop stops at 6 bits to keep the test fast: an 8-bit codebook requires ~1B exp()
      // evaluations (256 levels × 200 Lloyd-Max iterations × 10,000 Simpson intervals).
      // Structural correctness of compute(128, 8) is verified by
      // BasicProperties.eightBit_has256Levels.
      for (int bits = 1; bits <= 6; bits++) {
        LloydMaxCodebook codebook = LloydMaxCodebook.compute(dim, bits);
        double mse = 0;
        for (float s : samples) {
          float reconstructed = codebook.dequantize(codebook.quantize(s));
          float err = s - reconstructed;
          mse += err * err;
        }
        mse /= n;
        assertThat(mse)
            .as("MSE for %d bits should be less than %d bits", bits, bits - 1)
            .isLessThan(prevMse);
        prevMse = mse;
      }
    }

    @Test
    void lloydMax_betterThanUniformQuantizer_4bit() {
      int dim = 128;
      float sigma = 1f / (float) Math.sqrt(dim);
      Random rng = new Random(42L);

      int n = 10000;
      float[] samples = new float[n];
      for (int i = 0; i < n; i++) {
        samples[i] = (float) (rng.nextGaussian() * sigma);
      }

      // Lloyd-Max MSE
      LloydMaxCodebook codebook = LloydMaxCodebook.compute(dim, 4);
      double lloydMse = 0;
      for (float s : samples) {
        float r = codebook.dequantize(codebook.quantize(s));
        lloydMse += (s - r) * (s - r);
      }
      lloydMse /= n;

      // Uniform quantizer MSE (same range, 16 levels)
      float lo = -3.5f * sigma;
      float hi = 3.5f * sigma;
      double uniformMse = 0;
      for (float s : samples) {
        float clamped = Math.max(lo, Math.min(hi, s));
        int idx = Math.min(15, (int) ((clamped - lo) / (hi - lo) * 16));
        float reconstructed = lo + (idx + 0.5f) * (hi - lo) / 16f;
        uniformMse += (s - reconstructed) * (s - reconstructed);
      }
      uniformMse /= n;

      // Lloyd-Max should be at least as good as uniform (typically better)
      assertThat(lloydMse).isLessThanOrEqualTo(uniformMse);
    }
  }

  @Nested
  @Tag("unit")
  class DifferentDimensions {

    @Test
    void codebook_variesByDimension() {
      // Higher dimensions have smaller sigma, so centroids should be closer to 0
      LloydMaxCodebook cb64 = LloydMaxCodebook.compute(64, 4);
      LloydMaxCodebook cb768 = LloydMaxCodebook.compute(768, 4);

      float[] c64 = cb64.centroids();
      float[] c768 = cb768.centroids();

      // Max centroid for dim=768 should be smaller than for dim=64
      assertThat(Math.abs(c768[c768.length - 1])).isLessThan(Math.abs(c64[c64.length - 1]));
    }

    @Test
    void codebook_isDeterministic() {
      LloydMaxCodebook cb1 = LloydMaxCodebook.compute(128, 4);
      LloydMaxCodebook cb2 = LloydMaxCodebook.compute(128, 4);
      float[] c1 = cb1.centroids();
      float[] c2 = cb2.centroids();
      for (int i = 0; i < c1.length; i++) {
        assertThat(c1[i]).isEqualTo(c2[i]);
      }
    }

    @Test
    void compute_cacheReturnsSameInstance() {
      // LloydMaxCodebook.compute() caches results by (dimension, bits) to avoid
      // repeating the expensive Lloyd-Max iteration (~1B exp() calls for 8-bit).
      LloydMaxCodebook cb1 = LloydMaxCodebook.compute(256, 3);
      LloydMaxCodebook cb2 = LloydMaxCodebook.compute(256, 3);
      assertThat(cb1).isSameAs(cb2);
    }

    @Test
    void compute_differentParamsReturnDifferentInstances() {
      LloydMaxCodebook cb128_4 = LloydMaxCodebook.compute(128, 4);
      LloydMaxCodebook cb256_4 = LloydMaxCodebook.compute(256, 4);
      LloydMaxCodebook cb128_2 = LloydMaxCodebook.compute(128, 2);
      assertThat(cb128_4).isNotSameAs(cb256_4);
      assertThat(cb128_4).isNotSameAs(cb128_2);
    }
  }
}
