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

/**
 * NQT (Not-Quite Transcendental) nonlinear transform functions for NVQ. Uses IEEE 754 bit
 * manipulation for fast approximate logistic/logit computation without calling {@link Math#exp} or
 * {@link Math#log}.
 *
 * <p>The NQT logistic maps values through a scaled sigmoid curve, enabling per-subvector nonlinear
 * quantization that adapts to each vector's value distribution. The inverse (logit) enables
 * dequantization during scoring.
 *
 * <p><b>Key insight:</b> Standard logistic requires exp(), which is expensive. NQT uses IEEE 754
 * floating-point bit manipulation (extracting mantissa and exponent) to approximate exp() and log()
 * in ~4 FMA operations instead of ~20 for a Taylor series.
 *
 * @see NVQuantizer
 * @see NVQuantizedVectors
 */
final class NQTransform {

  /** Candidate alpha values for grid search optimization (geometric progression). */
  private static final float[] ALPHA_CANDIDATES = {
    0.1f, 0.15f, 0.2f, 0.3f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 7.0f, 10.0f, 15.0f, 20.0f,
    30.0f, 50.0f
  };

  private NQTransform() {}

  /**
   * NQT logistic approximation using IEEE 754 bit manipulation. Approximates {@code 1 / (1 +
   * exp(-z))} where {@code z = value * alpha - alpha * x0}.
   *
   * <p>The approximation works by decomposing exp(z) into 2^p * 2^f where p is the integer part and
   * f is the fractional part, using float bit manipulation to reconstruct 2^(p+f).
   *
   * @param value the input value
   * @param alpha the steepness parameter (controls sigmoid curvature)
   * @param x0 the midpoint parameter (sigmoid center)
   * @return approximate logistic(value * alpha - alpha * x0) in [0, 1]
   */
  static float logistic(float value, float alpha, float x0) {
    float z = Math.fma(value, alpha, -alpha * x0);
    // Clamp to avoid overflow in float→int conversion
    z = Math.max(-87f, Math.min(87f, z));
    int p = Math.round(z + 0.5f);
    int m = Float.floatToRawIntBits(Math.fma(z - p, 0.5f, 1f));
    float temp = Float.intBitsToFloat(m + (p << 23));
    return temp / (temp + 1f);
  }

  /**
   * NQT logit (inverse logistic) approximation using IEEE 754 bit manipulation. Approximates {@code
   * log(value / (1 - value)) * inverseAlpha + x0}.
   *
   * @param value the input value in (0, 1)
   * @param inverseAlpha 1/alpha
   * @param x0 the midpoint parameter
   * @return approximate logit(value) / alpha + x0
   */
  static float logit(float value, float inverseAlpha, float x0) {
    // Clamp to avoid division by zero or log of zero
    value = Math.max(1e-7f, Math.min(1f - 1e-7f, value));
    float z = value / (1f - value);
    int temp = Float.floatToRawIntBits(z);
    int e = temp & 0x7f800000;
    float pf = (float) ((e >> 23) - 128);
    float mf = Float.intBitsToFloat((temp & 0x007fffff) | 0x3f800000);
    return Math.fma(mf + pf, inverseAlpha, x0);
  }

  /**
   * Scaled logistic mapping [minVal, maxVal] → [0, 1]. Applies the NQT logistic after scaling alpha
   * and x0 to the subvector's value range.
   *
   * @param value the input value
   * @param scaledAlpha alpha / (maxVal - minVal)
   * @param scaledX0 (minVal + maxVal) / 2
   * @param logisticBias logistic(minVal, scaledAlpha, scaledX0)
   * @param logisticScale 1 / (logistic(maxVal, ...) - logistic(minVal, ...))
   * @return value mapped to [0, 1] via the scaled logistic
   */
  static float scaledLogistic(
      float value, float scaledAlpha, float scaledX0, float logisticBias, float logisticScale) {
    float raw = logistic(value, scaledAlpha, scaledX0);
    return (raw - logisticBias) * logisticScale;
  }

  /**
   * Inverse of scaled logistic: maps a quantized value back to the original value range.
   *
   * @param quantizedValue the quantized value in [0, 1] (typically byte/255.0)
   * @param inverseScaledAlpha 1 / scaledAlpha
   * @param scaledX0 (minVal + maxVal) / 2
   * @param logisticScale 1 / (logistic(maxVal) - logistic(minVal))
   * @param logisticBias logistic(minVal)
   * @return the dequantized value in [minVal, maxVal]
   */
  static float scaledLogit(
      float quantizedValue,
      float inverseScaledAlpha,
      float scaledX0,
      float logisticScale,
      float logisticBias) {
    float raw = quantizedValue / logisticScale + logisticBias;
    return logit(raw, inverseScaledAlpha, scaledX0);
  }

  /**
   * Grid-searches for the optimal alpha parameter that minimizes reconstruction error for a given
   * subvector. The midpoint x0 is always set to (minVal + maxVal) / 2.
   *
   * <p>For each candidate alpha, the reconstruction error is: {@code sum_d((sv[d] -
   * dequantize(quantize(sv[d])))²)}
   *
   * @param subvector the subvector values
   * @param minVal minimum value in the subvector
   * @param maxVal maximum value in the subvector
   * @return float[2]: [bestAlpha, bestX0]
   */
  static float[] optimizeTransform(float[] subvector, float minVal, float maxVal) {
    float x0 = (minVal + maxVal) / 2f;

    // If range is zero or tiny, use linear (alpha=1, effectively uniform quantization)
    float range = maxVal - minVal;
    if (range < 1e-10f) {
      return new float[] {1.0f, x0};
    }

    float bestAlpha = 1.0f;
    float bestError = computeReconstructionError(subvector, 1.0f, x0, minVal, maxVal);

    for (float alpha : ALPHA_CANDIDATES) {
      float error = computeReconstructionError(subvector, alpha, x0, minVal, maxVal);
      if (error < bestError) {
        bestError = error;
        bestAlpha = alpha;
      }
    }

    return new float[] {bestAlpha, x0};
  }

  /**
   * Computes the total squared reconstruction error for a given (alpha, x0) parameterization on the
   * given subvector. Quantizes each value to uint8 via the scaled logistic and then dequantizes
   * back, measuring the squared difference.
   */
  private static float computeReconstructionError(
      float[] subvector, float alpha, float x0, float minVal, float maxVal) {
    float range = maxVal - minVal;
    float scaledAlpha = alpha / range;
    float logMin = logistic(minVal, scaledAlpha, x0);
    float logMax = logistic(maxVal, scaledAlpha, x0);
    float logRange = logMax - logMin;

    if (logRange < 1e-10f) {
      // Degenerate: all values map to the same logistic output
      return Float.MAX_VALUE;
    }

    float logisticScale = 1f / logRange;
    float inverseScaledAlpha = range / alpha;

    float totalError = 0f;
    for (float v : subvector) {
      // Quantize
      float mapped = scaledLogistic(v, scaledAlpha, x0, logMin, logisticScale);
      int q = Math.round(mapped * 255f);
      q = Math.max(0, Math.min(255, q));

      // Dequantize
      float dequantized = scaledLogit(q / 255f, inverseScaledAlpha, x0, logisticScale, logMin);

      float diff = v - dequantized;
      totalError += diff * diff;
    }
    return totalError;
  }
}
