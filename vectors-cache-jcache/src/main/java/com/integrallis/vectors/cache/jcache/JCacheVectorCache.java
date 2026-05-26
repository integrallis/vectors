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
package com.integrallis.vectors.cache.jcache;

import com.integrallis.vectors.cache.CacheStats;
import com.integrallis.vectors.cache.VectorCache;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import javax.cache.Cache;

/**
 * Adapter that exposes any JSR-107 {@link Cache} as a {@link VectorCache}, so apps already using
 * Ehcache, Hazelcast, Infinispan, or Caffeine-JCache can plug into the vectors-cache SPI without
 * switching backends.
 *
 * <p>JCache statistics are provider-specific and typically enabled via provider configuration; to
 * keep the SPI independent of that, this wrapper tracks {@code hits} and {@code misses} itself via
 * {@link LongAdder} and reports {@code evictions=0} and {@code size=-1}. Providers that report
 * accurate sizes via {@code getCacheManager().getCache(...)} can swap in a richer implementation.
 */
public final class JCacheVectorCache<K, V> implements VectorCache<K, V> {

  private final Cache<K, V> cache;
  private final ConcurrentHashMap<K, Object> loadLocks = new ConcurrentHashMap<>();
  private final LongAdder hits = new LongAdder();
  private final LongAdder misses = new LongAdder();

  /**
   * @param cache non-null JSR-107 cache; caller retains ownership and is responsible for closing
   *     the underlying {@code CacheManager}
   */
  public JCacheVectorCache(Cache<K, V> cache) {
    this.cache = Objects.requireNonNull(cache, "cache");
  }

  @Override
  public Optional<V> get(K key) {
    V v = cache.get(Objects.requireNonNull(key, "key"));
    if (v == null) {
      misses.increment();
      return Optional.empty();
    }
    hits.increment();
    return Optional.of(v);
  }

  @Override
  public void put(K key, V value) {
    cache.put(Objects.requireNonNull(key, "key"), Objects.requireNonNull(value, "value"));
  }

  @Override
  public void invalidate(K key) {
    cache.remove(Objects.requireNonNull(key, "key"));
  }

  @Override
  public void invalidateAll() {
    cache.removeAll();
  }

  @Override
  public V getOrCompute(K key, Function<K, V> loader) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(loader, "loader");
    V existing = cache.get(key);
    if (existing != null) {
      hits.increment();
      return existing;
    }
    Object lock = loadLocks.computeIfAbsent(key, ignored -> new Object());
    try {
      synchronized (lock) {
        existing = cache.get(key);
        if (existing != null) {
          hits.increment();
          return existing;
        }
        misses.increment();
        V loaded = loader.apply(key);
        if (loaded == null) {
          throw new IllegalStateException("loader returned null for key " + key);
        }
        cache.put(key, loaded);
        return loaded;
      }
    } finally {
      loadLocks.remove(key, lock);
    }
  }

  @Override
  public CacheStats stats() {
    // Size -1 because JCache does not portably expose it; providers can extend this class.
    return new CacheStats(hits.sum(), misses.sum(), 0L, 0L, -1L);
  }

  @Override
  public void close() {
    // Intentional: caller owns the JSR-107 CacheManager lifecycle.
  }
}
