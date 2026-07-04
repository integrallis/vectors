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
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
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
 * <p><b>Eviction policy</b>: approximate LRU with a configurable maximum entry count, implemented
 * with a CLOCK (second-chance) sweep over a {@link ConcurrentHashMap}. Each entry carries a
 * reference bit that is set (lock-free) on every hit; eviction reclaims the first entry whose bit
 * is clear, giving the second-chance bit one full sweep to survive. This keeps the read path (cache
 * hits) completely lock-free — a hit is a {@code ConcurrentHashMap.get} plus a single volatile
 * write of the reference bit — instead of contending on the exclusive lock that an access-order
 * {@code LinkedHashMap} takes on <em>every</em> {@code get}.
 *
 * <p><b>Thread safety</b>: reads ({@link #get}) are lock-free. Structural mutations ({@link #put}
 * insertions and eviction, {@link #invalidateAll}) are serialized on a private write lock; the hot
 * read path never touches it.
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

  /** Backing store; {@code null} in bypass mode ({@code maxEntries == 0}). Reads are lock-free. */
  private final ConcurrentHashMap<QvCacheKey, Entry> store;

  /**
   * CLOCK ring of keys in insertion order. Holds exactly one entry per live key (a key is appended
   * only when a fresh entry is inserted, removed when it is evicted, and re-appended once when it
   * is granted a second chance). Only touched under {@link #writeLock}.
   */
  private final ConcurrentLinkedQueue<QvCacheKey> clock;

  /** Serializes structural mutations (insertion, eviction, clear). The read path never locks. */
  private final Object writeLock = new Object();

  private final AtomicLong hits = new AtomicLong();
  private final AtomicLong misses = new AtomicLong();
  private final AtomicLong evictions = new AtomicLong();

  /** Bypass instance used when caching is disabled ({@code maxEntries == 0}). */
  public static final QvCache DISABLED = new QvCache(0);

  /** Cache entry: the value plus a reference bit for the CLOCK second-chance sweep. */
  private static final class Entry {
    volatile SearchResult value;

    /** Set on every hit (lock-free); cleared by the eviction sweep to grant a second chance. */
    volatile boolean referenced;

    Entry(SearchResult value) {
      this.value = value;
    }
  }

  /**
   * Creates a QvCache with the given LRU capacity.
   *
   * @param maxEntries maximum number of cached entries; 0 means bypass (always miss)
   */
  public QvCache(int maxEntries) {
    this.maxEntries = maxEntries;
    if (maxEntries > 0) {
      this.store = new ConcurrentHashMap<>(maxEntries + 1, 0.75f);
      this.clock = new ConcurrentLinkedQueue<>();
    } else {
      this.store = null;
      this.clock = null;
    }
  }

  /** Returns {@code true} when caching is enabled ({@code maxEntries > 0}). */
  public boolean isEnabled() {
    return maxEntries > 0;
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
    return get(buildKey(query, k, filter));
  }

  /**
   * Looks up a cached result by a pre-built key. Lock-free: a {@link ConcurrentHashMap#get} plus,
   * on a hit, a single volatile write of the entry's reference bit.
   *
   * @param key the cache key (see {@link #buildKey})
   * @return the cached result, or {@link Optional#empty()} on a miss
   */
  public Optional<SearchResult> get(QvCacheKey key) {
    if (maxEntries == 0) return Optional.empty();
    Entry e = store.get(key);
    if (e != null) {
      // Second-chance bit — only write when unset to avoid needless cache-line churn.
      if (!e.referenced) e.referenced = true;
      hits.incrementAndGet();
      return Optional.of(e.value);
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
    put(buildKey(query, k, filter), result);
  }

  /**
   * Stores a result under a pre-built key. Structural mutation (insertion + CLOCK eviction) runs
   * under the write lock; updating the value of an already-present key is a lock-free volatile
   * write.
   *
   * @param key the cache key (see {@link #buildKey})
   * @param result the result to cache; must be immutable (records satisfy this)
   */
  public void put(QvCacheKey key, SearchResult result) {
    if (maxEntries == 0) return;
    Entry existing = store.get(key);
    if (existing != null) {
      existing.value = result;
      existing.referenced = true;
      return;
    }
    synchronized (writeLock) {
      Entry e = store.get(key);
      if (e != null) {
        e.value = result;
        e.referenced = true;
        return;
      }
      store.put(key, new Entry(result));
      clock.add(key);
      while (store.size() > maxEntries && evictOne()) {
        // keep evicting until the size bound is restored
      }
    }
  }

  /**
   * CLOCK second-chance eviction of a single entry. Must be called under {@link #writeLock}.
   *
   * <p>Sweeps the ring: an entry with its reference bit set is spared (bit cleared, re-queued); the
   * first entry with a clear bit is removed. Bounded by {@code 2 * size + 1} steps — after at most
   * one full sweep every bit is clear, so the next candidate is evictable.
   *
   * @return {@code true} if an entry was evicted
   */
  private boolean evictOne() {
    int budget = 2 * store.size() + 1;
    for (int i = 0; i < budget; i++) {
      QvCacheKey candidate = clock.poll();
      if (candidate == null) return false; // ring empty
      Entry e = store.get(candidate);
      if (e == null) continue; // stale ring slot (already removed) — drop it
      if (e.referenced) {
        e.referenced = false;
        clock.add(candidate); // second chance
      } else {
        store.remove(candidate, e);
        evictions.incrementAndGet();
        return true;
      }
    }
    return false;
  }

  /** Evicts all entries. Call after every {@code commit()} to prevent stale results. */
  public void invalidateAll() {
    if (maxEntries == 0) return;
    synchronized (writeLock) {
      store.clear();
      clock.clear();
    }
  }

  /** Returns a snapshot of cache statistics. */
  public CacheStats stats() {
    int size = maxEntries > 0 ? store.size() : 0;
    return new CacheStats(hits.get(), misses.get(), evictions.get(), size, maxEntries);
  }

  // ---------------------------------------------------------------------------
  // Key construction + query quantization
  // ---------------------------------------------------------------------------

  /**
   * Builds the cache key for a query. Exposed so callers can quantize + hash once and pass the
   * resulting key to both {@link #get(QvCacheKey)} and {@link #put(QvCacheKey, SearchResult)},
   * avoiding a redundant second quantize + filter-hash per search.
   *
   * @param query full-precision query vector
   * @param k number of results requested
   * @param filter filter predicate (may be {@code null})
   * @return the cache key
   */
  public QvCacheKey buildKey(float[] query, int k, Filter filter) {
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
