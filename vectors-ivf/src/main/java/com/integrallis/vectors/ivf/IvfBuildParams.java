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
 */
public record IvfBuildParams(
    int k, int maxIter, float gamma, boolean buildSoar, long seed, int harmonyKeyDims) {

  /** Validates parameters at construction time. */
  public IvfBuildParams {
    if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
    if (maxIter < 1) throw new IllegalArgumentException("maxIter must be >= 1, got " + maxIter);
    if (gamma < 0f) throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
    if (harmonyKeyDims < 0)
      throw new IllegalArgumentException("harmonyKeyDims must be >= 0, got " + harmonyKeyDims);
  }

  /**
   * Convenience factory: sensible defaults for a dataset of size {@code n}. HARMONY disabled
   * ({@link #harmonyKeyDims} = 0).
   */
  public static IvfBuildParams defaults(int n) {
    int k = Math.max(1, (int) Math.ceil(Math.sqrt(n)));
    return new IvfBuildParams(k, 50, 0.2f, true, 42L, 0);
  }

  /**
   * Returns a copy of these params with HARMONY enabled at {@code keyDims} key dimensions per
   * cluster.
   *
   * @param keyDims number of key dimensions to compute; must be &gt; 0
   */
  public IvfBuildParams withHarmony(int keyDims) {
    if (keyDims <= 0) throw new IllegalArgumentException("keyDims must be > 0, got " + keyDims);
    return new IvfBuildParams(k, maxIter, gamma, buildSoar, seed, keyDims);
  }
}
