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
package com.integrallis.vectors.demo.cache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * CI gate for the embedding-cache demo (audit T3.10). The demo's value proposition is "cached texts
 * don't reach the delegate model on the second call". This test pins that contract: across the
 * three phases (batch-with-dupes, mostly-hit batch, hot-key single embed), the delegate must be
 * called once per unique text and not once per request.
 */
class EmbeddingCacheAppTest {

  @Test
  void cacheServesRepeatedTextsWithoutInvokingTheDelegate() {
    EmbeddingCacheApp.DemoResult r = EmbeddingCacheApp.runDemo();

    // Phase 1: a batch of 6 segments (4 unique) — CachingEmbeddingModel's embedAll forwards the
    //          whole list opaquely, so all 6 hit the delegate; cache then holds 4 entries.
    // Phase 2: 3 segments (2 cached hits + 1 new "What is ANN?") — 1 delegate call.
    // Phase 3: 1 segment, already in cache — 0 delegate calls.
    // Total: 7 delegate invocations, 5 unique cache entries, 3 hits, 7 misses.
    assertThat(r.delegateCalls())
        .as(
            "phases 2+3 combined send 4 requests but should produce only 1 delegate call "
                + "(the new \"What is ANN?\"); observed %s",
            r.delegateCalls())
        .isEqualTo(7);
    assertThat(r.stats().size()).as("cache must hold all 5 distinct texts ever seen").isEqualTo(5L);
    assertThat(r.stats().hits())
        .as("phase 2 contributes 2 hits, phase 3 contributes 1 — totalling 3")
        .isEqualTo(3L);
    assertThat(r.stats().misses())
        .as("all phase-1 requests + the new phase-2 text miss the cache")
        .isEqualTo(7L);
    assertThat(r.stats().hitRate())
        .as("steady-state hit rate must be > 0.2 (3 hits / 10 lookups)")
        .isGreaterThan(0.2);
  }
}
