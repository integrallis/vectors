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
package com.integrallis.vectors.core;

import java.util.Locale;

/**
 * Arithmetic domain of the GGUF batched matmul entry points for Q4_K, Q6_K and Q8_0.
 *
 * <p>The process default is selected once with {@code -Dvectors.gguf.batchedMatmulKernel=integer|
 * band} (default {@code integer}) and is reported by {@link VectorRuntimeCapabilities
 * #ggufBatchedMatmulKernel()}. Explicit overloads taking this enum bypass the property.
 *
 * <p><b>Experimental.</b> {@link #BAND_F32} is an A/B arm, not a production default: it is not
 * numerically identical to {@link #INTEGER} (see the constant's documentation).
 */
public enum GgufBatchedMatmulKernel {
  /**
   * The established kernels: activations are quantized to Q8_K/Q8_0 in caller scratch and the dot
   * products run in integer lanes.
   */
  INTEGER,

  /**
   * Experimental dequantise-to-F32 band GEMM: a cache-sized panel of weight rows is dequantized to
   * F32 scratch and swept against the unquantized activations with a vector FMA register tile. The
   * caller's Q8 scratch arrays are validated but not written. Results equal the float dot product
   * of the dequantized weights with the activations, so they differ from {@link #INTEGER} by that
   * kernel's activation quantization error. Requires the Panama provider.
   */
  BAND_F32;

  /** System property selecting the process default. */
  public static final String PROPERTY = "vectors.gguf.batchedMatmulKernel";

  private static final GgufBatchedMatmulKernel CONFIGURED = parse(System.getProperty(PROPERTY));

  /** Returns the kernel requested by {@link #PROPERTY}, whether or not it can run here. */
  public static GgufBatchedMatmulKernel configured() {
    return CONFIGURED;
  }

  /**
   * Returns the kernel the default entry points actually run: the configured kernel, except that
   * {@link #BAND_F32} degrades to {@link #INTEGER} when the Vector API provider is not active (the
   * degradation is visible in {@link VectorRuntimeCapabilities#ggufBatchedMatmulKernel()}).
   */
  public static GgufBatchedMatmulKernel active() {
    return CONFIGURED == BAND_F32 && !VectorizationProvider.isPanamaEnabled()
        ? INTEGER
        : CONFIGURED;
  }

  static GgufBatchedMatmulKernel parse(String configured) {
    if (configured == null || configured.isBlank()) {
      return INTEGER;
    }
    return switch (configured.trim().toLowerCase(Locale.ROOT)) {
      case "integer", "int8" -> INTEGER;
      case "band", "band_f32", "band-f32" -> BAND_F32;
      default ->
          throw new IllegalArgumentException(
              "-D" + PROPERTY + " must be integer or band; got: " + configured);
    };
  }

  /**
   * Resolved configuration of the {@link #BAND_F32} arm (tile, column block, panel size, lanes)
   * whether or not it is the process default, so a caller selecting it explicitly can record what
   * ran; {@code "unavailable"} without the Vector API provider.
   */
  public static String bandConfiguration() {
    return VectorizationProvider.isPanamaEnabled() ? GgufBandGemm.describe() : "unavailable";
  }

  /** Human-readable description including the band arm's resolved tile, block and panel sizes. */
  static String describeActive() {
    if (active() == INTEGER) {
      return CONFIGURED == BAND_F32
          ? "integer (band requested; Vector API unavailable)"
          : "integer";
    }
    return GgufBandGemm.describe();
  }
}
