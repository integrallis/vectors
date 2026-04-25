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
package com.integrallis.vectors.ivf;

/**
 * Parameters for building an {@link IvfIndex}.
 *
 * @param k number of clusters; {@code ceil(sqrt(n))} is a common default
 * @param maxIter maximum k-means iterations (50 is typically sufficient)
 * @param gamma SOAR boundary expansion factor at build time (0.0 = disabled)
 * @param buildSoar whether to compute the SOAR spill map during training
 * @param seed RNG seed for reproducibility
 * @param harmonyKeyDims number of high-variance dimensions per cluster to pre-compute for HARMONY
 *     partial-distance pruning during EUCLIDEAN search (0 = disabled). A value in the range {@code
 *     [dim/8, dim/4]} is a practical starting point.
 * @param pqSubspaces number of PQ sub-vector partitions (M). {@code 0} disables product
 *     quantization and the index behaves as IVF-flat. Typical values: {@code dim/4}, {@code dim/8},
 *     or {@code dim/16} depending on target recall/memory tradeoff. Must evenly bound {@code dim}
 *     when non-zero; see {@link com.integrallis.vectors.quantization.ProductQuantizer#train}.
 * @param pqClusters centroids per subspace (Ks). Must be in {@code [1, 256]}; ignored when {@link
 *     #pqSubspaces} is 0. Default when PQ is enabled: 256.
 * @param pqAnisotropicThreshold ScaNN / AVQ anisotropic refinement threshold. Pass {@code -1f} for
 *     standard (unweighted) PQ. Typical non-negative values: 0.1–0.3. Ignored when PQ is disabled.
 */
public record IvfBuildParams(
    int k,
    int maxIter,
    float gamma,
    boolean buildSoar,
    long seed,
    int harmonyKeyDims,
    int pqSubspaces,
    int pqClusters,
    float pqAnisotropicThreshold) {

  /** Validates parameters at construction time. */
  public IvfBuildParams {
    if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
    if (maxIter < 1) throw new IllegalArgumentException("maxIter must be >= 1, got " + maxIter);
    if (gamma < 0f) throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
    if (harmonyKeyDims < 0)
      throw new IllegalArgumentException("harmonyKeyDims must be >= 0, got " + harmonyKeyDims);
    if (pqSubspaces < 0)
      throw new IllegalArgumentException("pqSubspaces must be >= 0, got " + pqSubspaces);
    if (pqSubspaces > 0 && (pqClusters < 1 || pqClusters > 256))
      throw new IllegalArgumentException(
          "pqClusters must be in [1, 256] when PQ is enabled, got " + pqClusters);
    if (pqAnisotropicThreshold != -1f
        && (pqAnisotropicThreshold < 0f || pqAnisotropicThreshold >= 1f))
      throw new IllegalArgumentException(
          "pqAnisotropicThreshold must be -1 (disabled) or in [0, 1), got "
              + pqAnisotropicThreshold);
  }

  /**
   * Backward-compatible 6-arg constructor: product quantization disabled. Existing IVF-flat callers
   * continue to compile unchanged.
   */
  public IvfBuildParams(
      int k, int maxIter, float gamma, boolean buildSoar, long seed, int harmonyKeyDims) {
    this(k, maxIter, gamma, buildSoar, seed, harmonyKeyDims, 0, 256, -1f);
  }

  /** Returns {@code true} when product quantization is enabled for this build. */
  public boolean pqEnabled() {
    return pqSubspaces > 0;
  }

  /**
   * Convenience factory: sensible defaults for a dataset of size {@code n}. HARMONY and PQ both
   * disabled.
   */
  public static IvfBuildParams defaults(int n) {
    int k = Math.max(1, (int) Math.ceil(Math.sqrt(n)));
    return new IvfBuildParams(k, 50, 0.2f, true, 42L, 0, 0, 256, -1f);
  }

  /**
   * Returns a copy of these params with HARMONY enabled at {@code keyDims} key dimensions per
   * cluster.
   *
   * @param keyDims number of key dimensions to compute; must be &gt; 0
   */
  public IvfBuildParams withHarmony(int keyDims) {
    if (keyDims <= 0) throw new IllegalArgumentException("keyDims must be > 0, got " + keyDims);
    return new IvfBuildParams(
        k,
        maxIter,
        gamma,
        buildSoar,
        seed,
        keyDims,
        pqSubspaces,
        pqClusters,
        pqAnisotropicThreshold);
  }

  /**
   * Returns a copy of these params with product quantization enabled at {@code subspaces}
   * sub-vector partitions and the default 256 centroids per subspace, unweighted (no anisotropic
   * refinement).
   *
   * @param subspaces number of PQ sub-vector partitions (M); must be &gt; 0
   */
  public IvfBuildParams withPq(int subspaces) {
    return withPq(subspaces, 256, -1f);
  }

  /**
   * Returns a copy of these params with product quantization enabled at the given subspace count,
   * cluster count, and anisotropic threshold.
   *
   * @param subspaces number of PQ sub-vector partitions (M); must be &gt; 0
   * @param clusters centroids per subspace (Ks); must be in {@code [1, 256]}
   * @param anisotropicThreshold pass {@code -1f} for unweighted PQ, else a value in {@code [0, 1)}
   */
  public IvfBuildParams withPq(int subspaces, int clusters, float anisotropicThreshold) {
    if (subspaces <= 0)
      throw new IllegalArgumentException("subspaces must be > 0, got " + subspaces);
    return new IvfBuildParams(
        k,
        maxIter,
        gamma,
        buildSoar,
        seed,
        harmonyKeyDims,
        subspaces,
        clusters,
        anisotropicThreshold);
  }
}
