package com.integrallis.vectors.cache;

/**
 * Lifetime counters for a {@link VectorCache} or {@link SemanticCache}. All fields are
 * monotonically-increasing since cache creation, except {@code size} which reflects the current
 * entry count.
 *
 * @param hits number of lookups that returned a cached value
 * @param misses number of lookups that did not find a cached value
 * @param evictions number of entries removed due to capacity / TTL / explicit invalidation
 * @param size current entry count
 */
public record CacheStats(long hits, long misses, long evictions, long size) {

  public static final CacheStats ZERO = new CacheStats(0, 0, 0, 0);

  /**
   * @return hits / (hits + misses); {@code 0.0} when there have been no requests
   */
  public double hitRate() {
    long total = hits + misses;
    return total == 0 ? 0.0 : (double) hits / total;
  }
}
