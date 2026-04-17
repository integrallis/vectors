package com.integrallis.vectors.ivf;

/**
 * Parameters for building an {@link IvfIndex}.
 *
 * @param k number of clusters; {@code ceil(sqrt(n))} is a common default
 * @param maxIter maximum k-means iterations (50 is typically sufficient)
 * @param gamma SOAR boundary expansion factor at build time (0.0 = disabled)
 * @param buildSoar whether to compute the SOAR spill map during training
 * @param seed RNG seed for reproducibility
 */
public record IvfBuildParams(int k, int maxIter, float gamma, boolean buildSoar, long seed) {

  /** Validates parameters at construction time. */
  public IvfBuildParams {
    if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
    if (maxIter < 1) throw new IllegalArgumentException("maxIter must be >= 1, got " + maxIter);
    if (gamma < 0f) throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
  }

  /** Convenience factory: sensible defaults for a dataset of size {@code n}. */
  public static IvfBuildParams defaults(int n) {
    int k = Math.max(1, (int) Math.ceil(Math.sqrt(n)));
    return new IvfBuildParams(k, 50, 0.2f, true, 42L);
  }
}
