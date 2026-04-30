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
    assertThat(s.rejections()).isZero();
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
