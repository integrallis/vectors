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

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;

/**
 * {@link VectorCache} that never stores anything. Useful in benchmarks that want to isolate the
 * cost of the loader, and in wiring code that needs a non-null default.
 */
public final class NoOpVectorCache<K, V> implements VectorCache<K, V> {

  private final LongAdder misses = new LongAdder();

  @Override
  public Optional<V> get(K key) {
    Objects.requireNonNull(key, "key");
    misses.increment();
    return Optional.empty();
  }

  @Override
  public void put(K key, V value) {
    // intentional no-op
  }

  @Override
  public void invalidate(K key) {
    // intentional no-op
  }

  @Override
  public void invalidateAll() {
    // intentional no-op
  }

  @Override
  public V getOrCompute(K key, Function<K, V> loader) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(loader, "loader");
    misses.increment();
    V value = loader.apply(key);
    if (value == null) {
      throw new IllegalStateException("loader returned null for key " + key);
    }
    return value;
  }

  @Override
  public CacheStats stats() {
    return new CacheStats(0L, misses.sum(), 0L, 0L, 0L);
  }
}
