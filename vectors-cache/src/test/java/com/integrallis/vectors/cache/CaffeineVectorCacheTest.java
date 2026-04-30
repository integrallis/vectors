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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CaffeineVectorCacheTest {

  @Test
  void putAndGetRoundTrip() {
    try (CaffeineVectorCache<String, float[]> cache =
        CaffeineVectorCache.<String, float[]>builder().build()) {
      float[] v = new float[] {1f, 2f, 3f};
      cache.put("k", v);
      assertThat(cache.get("k")).hasValue(v);
      assertThat(cache.get("missing")).isEmpty();
    }
  }

  @Test
  void invalidateRemovesEntry() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      cache.put("a", 1);
      assertThat(cache.get("a")).hasValue(1);
      cache.invalidate("a");
      assertThat(cache.get("a")).isEmpty();
    }
  }

  @Test
  void invalidateAllClears() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      cache.put("a", 1);
      cache.put("b", 2);
      cache.invalidateAll();
      assertThat(cache.stats().size()).isZero();
    }
  }

  @Test
  void statsTrackHitsAndMisses() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      cache.put("a", 1);
      assertThat(cache.get("a")).isPresent();
      assertThat(cache.get("b")).isEmpty();
      assertThat(cache.get("c")).isEmpty();
      CacheStats s = cache.stats();
      assertThat(s.hits()).isEqualTo(1);
      assertThat(s.misses()).isEqualTo(2);
      assertThat(s.rejections()).isZero();
      assertThat(s.hitRate()).isCloseTo(1.0 / 3.0, org.assertj.core.api.Assertions.within(1e-6));
    }
  }

  @Test
  void getOrComputeInvokesLoaderOnceAndCaches() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      AtomicInteger calls = new AtomicInteger();
      Integer a = cache.getOrCompute("k", k -> calls.incrementAndGet());
      Integer b = cache.getOrCompute("k", k -> calls.incrementAndGet());
      assertThat(a).isEqualTo(1);
      assertThat(b).isEqualTo(1);
      assertThat(calls.get()).isEqualTo(1);
    }
  }

  @Test
  void getOrComputeRejectsNullFromLoader() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      assertThatThrownBy(() -> cache.getOrCompute("k", k -> null))
          .isInstanceOf(IllegalStateException.class);
    }
  }

  @Test
  void maximumSizeEvicts() {
    CaffeineVectorCache<Integer, Integer> cache =
        CaffeineVectorCache.<Integer, Integer>builder().maximumSize(10).build();
    try {
      for (int i = 0; i < 100; i++) {
        cache.put(i, i);
      }
      // Caffeine eviction is asynchronous; close() delegates to cleanUp() which drains pending
      // maintenance tasks so the size assertion becomes deterministic.
      cache.close();
      assertThat(cache.stats().size()).isLessThanOrEqualTo(10);
    } finally {
      cache.close();
    }
  }

  @Test
  void expireAfterWriteDoesNotThrowOnBuild() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder()
            .expireAfterWrite(Duration.ofMinutes(5))
            .expireAfterAccess(Duration.ofMinutes(10))
            .build()) {
      cache.put("a", 1);
      assertThat(cache.get("a")).hasValue(1);
    }
  }

  @Test
  void putRejectsNullValue() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      assertThatThrownBy(() -> cache.put("a", null)).isInstanceOf(NullPointerException.class);
    }
  }

  @Test
  void putRejectsNullKey() {
    try (CaffeineVectorCache<String, Integer> cache =
        CaffeineVectorCache.<String, Integer>builder().build()) {
      assertThatThrownBy(() -> cache.put(null, 1)).isInstanceOf(NullPointerException.class);
    }
  }
}
