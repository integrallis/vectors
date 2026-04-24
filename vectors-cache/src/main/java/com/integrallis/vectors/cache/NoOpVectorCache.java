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
    return new CacheStats(0L, misses.sum(), 0L, 0L);
  }
}
