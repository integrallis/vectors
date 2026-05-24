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
 * Enumeration of the quantizer families supported by {@link QuantizationPipeline}.
 *
 * <p>Each constant maps to one of the concrete {@link Quantizer} implementations in this package.
 * The table below shows the default parameters used when a constant is selected without additional
 * builder overrides:
 *
 * <table>
 * <caption>Default quantizer parameters</caption>
 * <thead><tr><th>Type</th><th>Default bits/mode</th><th>Key builder override</th></tr></thead>
 * <tbody>
 * <tr><td>SCALAR_INT8</td><td>8-bit (INT8)</td><td>{@code .confidence(float)}</td></tr>
 * <tr><td>SCALAR_INT4</td><td>4-bit (INT4)</td><td>{@code .confidence(float)}</td></tr>
 * <tr><td>PRODUCT</td><td>8 subspaces, 256 clusters</td><td>{@code .subspaces(int) .clusters(int)}</td></tr>
 * <tr><td>OPTIMIZED_PRODUCT</td><td>8 subspaces, 256 clusters, 5 iters</td><td>{@code .subspaces(int) .iterations(int)}</td></tr>
 * <tr><td>BINARY_SIGN</td><td>sign-bit (1-bit, no centroid)</td><td>—</td></tr>
 * <tr><td>BINARY_BBQ</td><td>BBQ (1-bit + centroid)</td><td>—</td></tr>
 * <tr><td>RABIT</td><td>1-bit with random rotation</td><td>{@code .seed(long) .rotation(Rotation)}</td></tr>
 * <tr><td>EXTENDED_RABIT</td><td>4-bit multi-layer RaBitQ</td><td>{@code .bits(int) .seed(long)}</td></tr>
 * <tr><td>TURBO</td><td>4-bit MSE-optimal per-coordinate</td><td>{@code .bits(int) .seed(long)}</td></tr>
 * <tr><td>NVQ</td><td>default subvectors = dim/64</td><td>{@code .subspaces(int)}</td></tr>
 * </tbody>
 * </table>
 */
public enum QuantizerType {

  /** Scalar quantization to 8-bit integers (INT8). */
  SCALAR_INT8,

  /** Scalar quantization to 4-bit integers (INT4). */
  SCALAR_INT4,

  /** Product quantization (PQ): partitions the vector space into subspaces and clusters each. */
  PRODUCT,

  /**
   * Optimized Product Quantization (OPQ): iteratively learns a rotation that aligns the data with
   * PQ subspace boundaries, improving recall vs standard PQ.
   */
  OPTIMIZED_PRODUCT,

  /**
   * Binary quantization using the sign bit (1 bit/dimension, no centroid subtraction). Cheapest
   * compression; best for pre-normalized vectors.
   */
  BINARY_SIGN,

  /**
   * Balanced Binary Quantization (BBQ): subtracts the dataset centroid before sign-binarization,
   * improving recall for non-zero-mean distributions.
   */
  BINARY_BBQ,

  /**
   * RaBitQ 1-bit quantization with a learned random rotation. Provides error-correction codes for
   * better recall than plain sign binarization.
   */
  RABIT,

  /**
   * Extended RaBitQ: multi-bit (2–8) extension of RaBitQ for a precision–speed tradeoff between
   * RABIT and SCALAR_INT8.
   */
  EXTENDED_RABIT,

  /**
   * TurboQuantizer: MSE-optimal per-coordinate scalar quantization after a rotation. Achieves
   * better quality than uniform scalar quantization at the same bit width.
   */
  TURBO,

  /**
   * Non-uniform Vector Quantization (NVQ): partitions dimensions into variable-size subvectors
   * aligned to the data distribution.
   */
  NVQ,
}
