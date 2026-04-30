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

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

/**
 * {@link VectorCache} backed by a Caffeine {@code Cache}. Configure via the {@link Builder}. By
 * default the cache is bounded at 10 000 entries with no TTL; callers typically tighten both for
 * embedding workloads.
 */
public final class CaffeineVectorCache<K, V> implements VectorCache<K, V> {

  private final Cache<K, V> cache;

  private CaffeineVectorCache(Cache<K, V> cache) {
    this.cache = Objects.requireNonNull(cache, "cache");
  }

  public static <K, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  @Override
  public Optional<V> get(K key) {
    return Optional.ofNullable(cache.getIfPresent(Objects.requireNonNull(key, "key")));
  }

  @Override
  public void put(K key, V value) {
    cache.put(Objects.requireNonNull(key, "key"), Objects.requireNonNull(value, "value"));
  }

  @Override
  public void invalidate(K key) {
    cache.invalidate(Objects.requireNonNull(key, "key"));
  }

  @Override
  public void invalidateAll() {
    cache.invalidateAll();
  }

  @Override
  public V getOrCompute(K key, Function<K, V> loader) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(loader, "loader");
    return cache.get(
        key,
        k -> {
          V value = loader.apply(k);
          if (value == null) {
            throw new IllegalStateException("loader returned null for key " + k);
          }
          return value;
        });
  }

  @Override
  public CacheStats stats() {
    com.github.benmanes.caffeine.cache.stats.CacheStats cs = cache.stats();
    return new CacheStats(
        cs.hitCount(), cs.missCount(), cs.evictionCount(), 0L, cache.estimatedSize());
  }

  @Override
  public void close() {
    cache.cleanUp();
  }

  /** Fluent builder. Stats are always enabled so {@link #stats()} returns real counters. */
  public static final class Builder<K, V> {
    private long maximumSize = 10_000L;
    private Duration expireAfterWrite;
    private Duration expireAfterAccess;

    private Builder() {}

    public Builder<K, V> maximumSize(long n) {
      if (n < 0) {
        throw new IllegalArgumentException("maximumSize must be non-negative");
      }
      this.maximumSize = n;
      return this;
    }

    public Builder<K, V> expireAfterWrite(Duration d) {
      this.expireAfterWrite = Objects.requireNonNull(d, "duration");
      return this;
    }

    public Builder<K, V> expireAfterAccess(Duration d) {
      this.expireAfterAccess = Objects.requireNonNull(d, "duration");
      return this;
    }

    public CaffeineVectorCache<K, V> build() {
      Caffeine<Object, Object> b = Caffeine.newBuilder().maximumSize(maximumSize).recordStats();
      if (expireAfterWrite != null) {
        b.expireAfterWrite(expireAfterWrite);
      }
      if (expireAfterAccess != null) {
        b.expireAfterAccess(expireAfterAccess);
      }
      return new CaffeineVectorCache<>(b.build());
    }
  }
}
