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

import java.util.Optional;
import java.util.function.Function;

/**
 * Thin SPI for exact-match, key/value caching. Inspired by JCache without inheriting its runtime
 * machinery so implementers stay free to delegate to Caffeine, Redis, Hazelcast, or a null sink.
 *
 * <p>Implementations must be safe for concurrent access. {@link #getOrCompute(Object, Function)} is
 * the hot path for embedding caches: a miss triggers the loader at most once per key under
 * contention when the backing store supports it.
 *
 * @param <K> key type — often {@code String} (a canonical hash of an embedding input)
 * @param <V> value type — typically {@code float[]} for embedding caches
 */
public interface VectorCache<K, V> extends AutoCloseable {

  /**
   * @param key non-null lookup key
   * @return the cached value, or empty if absent / expired
   */
  Optional<V> get(K key);

  /**
   * Puts or replaces the value associated with {@code key}. Null values are rejected to keep the
   * contract unambiguous; callers wishing to record negative results should use a sentinel.
   */
  void put(K key, V value);

  /** Removes the entry, if any. */
  void invalidate(K key);

  /** Removes every entry. */
  void invalidateAll();

  /** Snapshot of the lifetime counters. Never returns {@code null}. */
  CacheStats stats();

  /**
   * Atomic lookup-or-compute. Default implementation is not atomic; Caffeine-backed caches override
   * with a single-flight loader so the {@code loader} runs once per key under contention.
   *
   * @param key non-null lookup key
   * @param loader invoked on miss; must return a non-null value
   * @return cached or freshly-loaded value
   */
  default V getOrCompute(K key, Function<K, V> loader) {
    return get(key)
        .orElseGet(
            () -> {
              V value = loader.apply(key);
              if (value == null) {
                throw new IllegalStateException("loader returned null for key " + key);
              }
              put(key, value);
              return value;
            });
  }

  /** Closes any underlying resources; default is no-op. */
  @Override
  default void close() {
    // no-op by default; Caffeine has nothing to release, Redis clients do.
  }
}
