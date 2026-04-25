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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CacheStats;
import java.util.concurrent.atomic.AtomicInteger;
import javax.cache.Cache;
import javax.cache.CacheManager;
import javax.cache.Caching;
import javax.cache.configuration.MutableConfiguration;
import javax.cache.spi.CachingProvider;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JCacheVectorCacheTest {

  private CachingProvider provider;
  private CacheManager manager;
  private Cache<String, Integer> backing;
  private JCacheVectorCache<String, Integer> cache;

  @BeforeEach
  void setUp() {
    provider =
        Caching.getCachingProvider(
            "com.github.benmanes.caffeine.jcache.spi.CaffeineCachingProvider");
    manager = provider.getCacheManager();
    manager.destroyCache("test");
    backing =
        manager.createCache(
            "test",
            new MutableConfiguration<String, Integer>()
                .setTypes(String.class, Integer.class)
                .setStatisticsEnabled(false));
    cache = new JCacheVectorCache<>(backing);
  }

  @AfterEach
  void tearDown() {
    manager.destroyCache("test");
    manager.close();
    provider.close();
  }

  @Test
  void putAndGetRoundTrip() {
    cache.put("a", 1);
    assertThat(cache.get("a")).hasValue(1);
    assertThat(cache.get("missing")).isEmpty();
  }

  @Test
  void invalidateRemovesEntry() {
    cache.put("a", 1);
    cache.invalidate("a");
    assertThat(cache.get("a")).isEmpty();
  }

  @Test
  void invalidateAllClears() {
    cache.put("a", 1);
    cache.put("b", 2);
    cache.invalidateAll();
    assertThat(cache.get("a")).isEmpty();
    assertThat(cache.get("b")).isEmpty();
  }

  @Test
  void statsTrackHitsAndMisses() {
    cache.put("a", 1);
    assertThat(cache.get("a")).isPresent();
    assertThat(cache.get("b")).isEmpty();
    CacheStats s = cache.stats();
    assertThat(s.hits()).isEqualTo(1);
    assertThat(s.misses()).isEqualTo(1);
    assertThat(s.size()).isEqualTo(-1L);
  }

  @Test
  void getOrComputeInvokesLoaderOnce() {
    AtomicInteger calls = new AtomicInteger();
    Integer a = cache.getOrCompute("k", k -> calls.incrementAndGet());
    Integer b = cache.getOrCompute("k", k -> calls.incrementAndGet());
    assertThat(a).isEqualTo(1);
    assertThat(b).isEqualTo(1);
    assertThat(calls.get()).isEqualTo(1);
  }
}
