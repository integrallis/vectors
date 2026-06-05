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

import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;

/**
 * Quantized Johnson-Lindenstrauss (QJL) sketch — the second stage of TurboQuant's unbiased
 * inner-product quantizer (TurboQuant_prod, arXiv:2504.19874; QJL is AAAI 2025). A single
 * data-oblivious {@code d×d} Gaussian sketch {@code S} (rows {@code g_i ~ N(0, I_d)}) is shared
 * across all vectors and regenerated deterministically from a seed (so only the seed is persisted).
 *
 * <p>Encoding a residual {@code r} stores only the {@code d} sign bits {@code sign(S·r)}. The
 * unbiased reconstruction of {@code r}'s direction is
 *
 * <pre>{@code   r ≈ sqrt(pi/2)/d · Sᵀ · sign(S·r)   (then scaled by the stored ||r||)}</pre>
 *
 * because {@code E[ sqrt(pi/2)/d · Sᵀ sign(S·u) ] = u} for a unit vector {@code u}. Adding this
 * term to the first-stage MSE reconstruction removes the multiplicative bias that an MSE-only
 * quantizer has for inner products (the paper notes MSE-only is "unsuitable for
 * inner-product-dependent applications").
 *
 * <p>All operations are in the padded, rotated coordinate space (dimension = padded dimension).
 */
final class QjlSketch {

  private final int dimension;
  private final long seed;
  private final float[][] s; // s[i] = row g_i, length dimension
  private final float invScale; // sqrt(pi/2) / dimension

  private QjlSketch(int dimension, long seed, float[][] s) {
    this.dimension = dimension;
    this.seed = seed;
    this.s = s;
    this.invScale = (float) (Math.sqrt(Math.PI / 2.0) / dimension);
  }

  /**
   * Generates a {@code dimension×dimension} Gaussian sketch deterministically from {@code seed}.
   *
   * @param dimension the (padded) working dimension
   * @param seed RNG seed; the same seed reproduces the same sketch
   */
  static QjlSketch generate(int dimension, long seed) {
    Random rng = new Random(seed);
    float[][] s = new float[dimension][dimension];
    for (int i = 0; i < dimension; i++) {
      float[] row = s[i];
      for (int j = 0; j < dimension; j++) {
        row[j] = (float) rng.nextGaussian();
      }
    }
    return new QjlSketch(dimension, seed, s);
  }

  /** The seed this sketch was generated from (persisted by the codec; regenerable). */
  long seed() {
    return seed;
  }

  /** The working (padded) dimension. */
  int dimension() {
    return dimension;
  }

  /** Number of bytes needed to store the {@code dimension} sign bits. */
  int bitBytes() {
    return (dimension + 7) / 8;
  }

  /**
   * Encodes {@code sign(S·r)} into a freshly allocated, LSB-first packed bit array. Bit {@code i}
   * is 1 iff {@code ⟨g_i, r⟩ >= 0}.
   *
   * @param r the residual vector in rotated space (length {@code dimension})
   */
  byte[] signBits(float[] r) {
    byte[] bits = new byte[bitBytes()];
    for (int i = 0; i < dimension; i++) {
      if (VectorUtil.dotProduct(s[i], r) >= 0f) {
        bits[i >> 3] |= (byte) (1 << (i & 7));
      }
    }
    return bits;
  }

  /**
   * Adds the unbiased QJL residual estimate {@code sqrt(pi/2)/d · gammaR · Sᵀ·sign} into {@code
   * out} (in place), where {@code sign_i = +1} if bit {@code i} is set else {@code -1}.
   *
   * @param bits packed sign bits from {@link #signBits}
   * @param gammaR the stored residual magnitude {@code ||r||}
   * @param out accumulator of length {@code dimension} (typically the first-stage reconstruction)
   */
  void addInverseResidual(byte[] bits, float gammaR, float[] out) {
    float scale = invScale * gammaR;
    if (scale == 0f) {
      return;
    }
    for (int i = 0; i < dimension; i++) {
      float signScaled = ((bits[i >> 3] >> (i & 7)) & 1) != 0 ? scale : -scale;
      float[] row = s[i];
      for (int j = 0; j < dimension; j++) {
        out[j] += row[j] * signScaled;
      }
    }
  }
}
