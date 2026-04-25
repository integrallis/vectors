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
package com.integrallis.vectors.db;

/**
 * Build-time quantizer parameters. These control how the quantizer is trained at commit time and
 * are NOT persisted on disk — the trained quantizer state is serialized in {@code quantized.bin}
 * and captures everything needed to reconstruct the quantizer without the original build params.
 *
 * <p>Each record corresponds to one or more {@link QuantizerKind} values:
 *
 * <ul>
 *   <li>{@link ScalarParams} — {@link QuantizerKind#SQ8} and {@link QuantizerKind#SQ4}. No extra
 *       parameters; the bit-width is determined by the kind.
 *   <li>{@link PqParams} — {@link QuantizerKind#PQ}.
 *   <li>{@link BqParams} — {@link QuantizerKind#BQ}.
 *   <li>{@link RaBitParams} — {@link QuantizerKind#RABITQ}.
 *   <li>{@link NvqParams} — {@link QuantizerKind#NVQ}.
 * </ul>
 *
 * <p>When {@link QuantizerKind#NONE}, the params field must be {@code null}. When non-NONE, the
 * params may be {@code null} to use defaults, or a record whose type must match the kind.
 */
public sealed interface QuantizerParams {

  /**
   * Parameters for scalar quantization ({@link QuantizerKind#SQ8} and {@link QuantizerKind#SQ4}).
   * Currently empty — the bit-width is inferred from the kind, and confidence-interval quantiles
   * are computed automatically during training.
   */
  record ScalarParams() implements QuantizerParams {}

  /**
   * Parameters for product quantization ({@link QuantizerKind#PQ}).
   *
   * @param numSubspaces number of subspaces (M). Must be positive and must evenly divide the vector
   *     dimension. Default: {@code max(1, dimension / 8)}.
   * @param numClusters number of clusters per subspace (Ks). Must be in [2, 256]. Default: 256.
   * @param center whether to subtract a global centroid before quantization. Default: true.
   * @param trainThreads worker thread count for per-subspace k-means during training. Must be
   *     {@code >= 1}. Default: 1 (sequential, byte-identical to pre-R2.E releases). Values {@code >
   *     1} route through the parallel path of {@link
   *     com.integrallis.vectors.quantization.ProductQuantizer#train(com.integrallis.vectors.quantization.VectorDataset,
   *     int, int, boolean, int)} and yield a deterministic but numerically distinct codebook.
   */
  record PqParams(int numSubspaces, int numClusters, boolean center, int trainThreads)
      implements QuantizerParams {
    public PqParams {
      if (numSubspaces <= 0) {
        throw new IllegalArgumentException("numSubspaces must be positive: " + numSubspaces);
      }
      if (numClusters < 2 || numClusters > 256) {
        throw new IllegalArgumentException("numClusters must be in [2, 256]: " + numClusters);
      }
      if (trainThreads < 1) {
        throw new IllegalArgumentException("trainThreads must be >= 1: " + trainThreads);
      }
    }

    /** Convenience constructor defaulting to single-threaded training. */
    public PqParams(int numSubspaces, int numClusters, boolean center) {
      this(numSubspaces, numClusters, center, 1);
    }
  }

  /**
   * Parameters for binary quantization ({@link QuantizerKind#BQ}).
   *
   * @param bbq whether to use BBQ mode (true) or plain sign-bit mode (false). BBQ computes a
   *     centroid and per-vector correction factors for asymmetric distance estimation. Default:
   *     true (BBQ is strictly more accurate than sign-bit for distance estimation).
   */
  record BqParams(boolean bbq) implements QuantizerParams {}

  /**
   * Parameters for RaBitQ ({@link QuantizerKind#RABITQ}).
   *
   * @param seed random seed for the rotation matrix. Default: 42L for reproducibility.
   */
  record RaBitParams(long seed) implements QuantizerParams {}

  /**
   * Parameters for NVQ ({@link QuantizerKind#NVQ}).
   *
   * @param numSubvectors number of subvectors (M). Must be positive and must evenly divide the
   *     vector dimension. Default: {@code max(1, dimension / 4)}.
   */
  record NvqParams(int numSubvectors) implements QuantizerParams {
    public NvqParams {
      if (numSubvectors <= 0) {
        throw new IllegalArgumentException("numSubvectors must be positive: " + numSubvectors);
      }
    }
  }
}
