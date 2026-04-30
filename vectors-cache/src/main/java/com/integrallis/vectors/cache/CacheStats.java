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
package com.integrallis.vectors.cache;

/**
 * Lifetime counters for a {@link VectorCache} or {@link SemanticCache}. All fields are
 * monotonically-increasing since cache creation, except {@code size} which reflects the current
 * entry count.
 *
 * @param hits number of lookups that returned a cached value
 * @param misses number of lookups that did not find a cached value
 * @param evictions number of entries removed due to capacity / TTL / explicit invalidation
 * @param rejections number of {@code put} calls rejected by the {@link CacheAdmissionPolicy}
 * @param size current entry count
 */
public record CacheStats(long hits, long misses, long evictions, long rejections, long size) {

  public static final CacheStats ZERO = new CacheStats(0, 0, 0, 0, 0);

  /**
   * @return hits / (hits + misses); {@code 0.0} when there have been no requests
   */
  public double hitRate() {
    long total = hits + misses;
    return total == 0 ? 0.0 : (double) hits / total;
  }
}
