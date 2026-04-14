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

  /** Convenience factory with SOAR disabled and no score threshold. */
  public static IvfSearchRequest of(float[] query, int k, int nprobe) {
    return new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE);
  }
}
