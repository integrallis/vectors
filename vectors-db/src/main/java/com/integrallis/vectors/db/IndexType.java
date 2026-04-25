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
 * Index backend selector. Step 2 only wires {@link #FLAT}; other values are reserved for subsequent
 * steps and throw {@link UnsupportedOperationException} at build time.
 */
public enum IndexType {
  /** Brute-force linear scan. Reference implementation, always available. */
  FLAT,

  /** Hierarchical Navigable Small World (HNSW) graph index. Deferred to Step 4. */
  HNSW,

  /** Vamana/DiskANN graph index. Deferred to Step 6. */
  VAMANA,

  /** Inverted-file (flat) clustering index. Deferred to a later step. */
  IVF_FLAT,

  /**
   * Inverted-file clustering index with product-quantised posting lists (IVF-PQ). Encodes each
   * vector as an {@code M}-byte code of its per-cluster residual and scores probed partitions via
   * asymmetric distance computation (ADC) against the raw query. An optional full-precision rescore
   * pass (see {@code ivfRescoreFactor}) trades extra scoring work for recall.
   */
  IVF_PQ,

  /**
   * GPU-accelerated brute-force search via NVIDIA cuVS. Requires {@code libcuvs.so} and a
   * compatible CUDA device at runtime; build fails with {@code GpuUnavailableException} otherwise.
   * In-memory only; persistent mode is deferred.
   */
  CUVS_BRUTEFORCE,

  /**
   * GPU-accelerated CAGRA graph index via NVIDIA cuVS. Requires {@code libcuvs.so} and a compatible
   * CUDA device at runtime; build fails with {@code GpuUnavailableException} otherwise. In-memory
   * only; persistent mode is deferred.
   */
  CUVS_CAGRA
}
