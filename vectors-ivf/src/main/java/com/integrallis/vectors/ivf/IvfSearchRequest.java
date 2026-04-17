package com.integrallis.vectors.ivf;

/**
 * Search parameters for {@link IvfIndex#search}.
 *
 * @param query query vector; must have the same dimension as the index
 * @param k number of top results to return
 * @param nprobe number of clusters to probe (higher = better recall, slower)
 * @param gamma SOAR boundary expansion at search time (0.0 = disabled)
 * @param minScore minimum score threshold; hits below this score are excluded
 */
public record IvfSearchRequest(float[] query, int k, int nprobe, float gamma, float minScore) {

  /** Validates parameters at construction time. */
  public IvfSearchRequest {
    if (query == null) throw new IllegalArgumentException("query must not be null");
    if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
    if (nprobe < 1) throw new IllegalArgumentException("nprobe must be >= 1, got " + nprobe);
    if (gamma < 0f) throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
  }

  /** Convenience factory with SOAR disabled and no score threshold. */
  public static IvfSearchRequest of(float[] query, int k, int nprobe) {
    return new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE);
  }
}
