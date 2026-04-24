package com.integrallis.vectors.cache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class NoOpVectorCacheTest {

  @Test
  void getAlwaysMisses() {
    NoOpVectorCache<String, Integer> cache = new NoOpVectorCache<>();
    cache.put("a", 1);
    assertThat(cache.get("a")).isEmpty();
    CacheStats s = cache.stats();
    assertThat(s.hits()).isZero();
    assertThat(s.misses()).isEqualTo(1);
  }

  @Test
  void getOrComputeInvokesLoaderEveryTime() {
    NoOpVectorCache<String, Integer> cache = new NoOpVectorCache<>();
    AtomicInteger calls = new AtomicInteger();
    cache.getOrCompute("k", k -> calls.incrementAndGet());
    cache.getOrCompute("k", k -> calls.incrementAndGet());
    cache.getOrCompute("k", k -> calls.incrementAndGet());
    assertThat(calls.get()).isEqualTo(3);
  }

  @Test
  void statsAlwaysReportZeroSize() {
    NoOpVectorCache<String, Integer> cache = new NoOpVectorCache<>();
    cache.put("a", 1);
    cache.put("b", 2);
    assertThat(cache.stats().size()).isZero();
  }
}
