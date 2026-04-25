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
package com.integrallis.vectors.db.cache;

import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.db.SearchResult;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Quantized-Vector Query Result Cache (QVCache).
 *
 * <p>Caches {@link SearchResult} objects keyed by a {@link QvCacheKey} that combines:
 *
 * <ul>
 *   <li>a scalar int8 quantization of the query vector (max-abs normalization)
 *   <li>the requested result count {@code k}
 *   <li>a stable hash of the {@link Filter} predicate
 * </ul>
 *
 * <p><b>Eviction policy</b>: LRU with a configurable maximum entry count.
 *
 * <p><b>Thread safety</b>: all operations are synchronized on the internal map instance.
 *
 * <p><b>Invalidation</b>: call {@link #invalidateAll()} after every committed write. This ensures
 * that the cache never serves stale results after the underlying index has changed.
 *
 * <p>When {@code maxEntries} is 0 the cache is in bypass mode — {@link #get} always returns empty
 * and {@link #put} is a no-op. This allows callers to construct the cache unconditionally and
 * enable/disable it purely through configuration.
 */
public final class QvCache {

  private final int maxEntries;
  private final Map<QvCacheKey, SearchResult> store;
  private final AtomicLong hits = new AtomicLong();
  private final AtomicLong misses = new AtomicLong();
  private final AtomicLong evictions = new AtomicLong();

  /** Bypass instance used when caching is disabled ({@code maxEntries == 0}). */
  public static final QvCache DISABLED = new QvCache(0);

  /**
   * Creates a QvCache with the given LRU capacity.
   *
   * @param maxEntries maximum number of cached entries; 0 means bypass (always miss)
   */
  public QvCache(int maxEntries) {
    this.maxEntries = maxEntries;
    if (maxEntries > 0) {
      this.store =
          Collections.synchronizedMap(
              new LinkedHashMap<>(maxEntries + 1, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<QvCacheKey, SearchResult> eldest) {
                  boolean remove = size() > maxEntries;
                  if (remove) evictions.incrementAndGet();
                  return remove;
                }
              });
    } else {
      this.store = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Looks up a cached result for the given query parameters.
   *
   * @param query full-precision query vector
   * @param k number of results requested
   * @param filter filter predicate (may be {@code null})
   * @return the cached result, or {@link Optional#empty()} on a miss
   */
  public Optional<SearchResult> get(float[] query, int k, Filter filter) {
    if (maxEntries == 0) return Optional.empty();
    QvCacheKey key = buildKey(query, k, filter);
    SearchResult cached = store.get(key);
    if (cached != null) {
      hits.incrementAndGet();
      return Optional.of(cached);
    }
    misses.incrementAndGet();
    return Optional.empty();
  }

  /**
   * Stores a result in the cache.
   *
   * @param query full-precision query vector
   * @param k number of results requested
   * @param filter filter predicate (may be {@code null})
   * @param result the result to cache; must be immutable (records satisfy this)
   */
  public void put(float[] query, int k, Filter filter, SearchResult result) {
    if (maxEntries == 0) return;
    QvCacheKey key = buildKey(query, k, filter);
    store.put(key, result);
  }

  /** Evicts all entries. Call after every {@code commit()} to prevent stale results. */
  public void invalidateAll() {
    if (maxEntries == 0) return;
    store.clear();
  }

  /** Returns a snapshot of cache statistics. */
  public CacheStats stats() {
    int size = maxEntries > 0 ? store.size() : 0;
    return new CacheStats(hits.get(), misses.get(), evictions.get(), size, maxEntries);
  }

  // ---------------------------------------------------------------------------
  // Key construction + query quantization
  // ---------------------------------------------------------------------------

  private QvCacheKey buildKey(float[] query, int k, Filter filter) {
    byte[] quantized = quantize(query);
    long filterHash = FilterHasher.hash(filter);
    return new QvCacheKey(quantized, k, filterHash);
  }

  /**
   * Quantizes a float query vector to signed int8 (one byte per dimension) using max-absolute-value
   * normalization. Vectors that are proportionally identical map to the same byte array, grouping
   * semantically similar queries under the same cache key.
   */
  public static byte[] quantize(float[] v) {
    // Find max absolute value for normalizing to [-127, 127]
    float maxAbs = 1e-9f; // avoid division by zero
    for (float f : v) {
      float a = Math.abs(f);
      if (a > maxAbs) maxAbs = a;
    }
    byte[] out = new byte[v.length];
    float scale = 127.0f / maxAbs;
    for (int i = 0; i < v.length; i++) {
      out[i] = (byte) Math.round(v[i] * scale);
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Nested stats record
  // ---------------------------------------------------------------------------

  /**
   * Snapshot of cache performance counters.
   *
   * @param hits number of cache hits since creation or last reset
   * @param misses number of cache misses since creation or last reset
   * @param evictions number of LRU evictions since creation or last reset
   * @param size current number of cached entries
   * @param maxSize configured LRU capacity (0 = cache disabled)
   */
  public record CacheStats(long hits, long misses, long evictions, int size, int maxSize) {

    /** Hit ratio in {@code [0.0, 1.0]}; returns 0.0 when no requests have been made. */
    public double hitRatio() {
      long total = hits + misses;
      return total == 0 ? 0.0 : (double) hits / total;
    }
  }
}
