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
package com.integrallis.vectors.demo.semanticcache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * CI gate for the semantic-cache demo (audit T3.10). The demo's value proposition is "cached
 * answers serve paraphrased queries when their similarity clears the threshold, and only novel
 * queries reach the real LLM". Pins the contract: at least the two exact-match queries hit, the
 * "What is scalar quantization?" novel query misses, and the misses+LLM-calls accounting agrees.
 */
class SemanticCacheAppTest {

  @Test
  void exactMatchesHitAndNovelQueriesMiss() {
    SemanticCacheApp.DemoResult r = SemanticCacheApp.runDemo();

    int total = r.hits() + r.misses();
    assertThat(total).as("demo issues exactly 6 paraphrased queries").isEqualTo(6);

    // Two queries are byte-exact matches of seeded canonical questions ("What is HNSW?" and
    // "Explain product quantization."). They must hit unconditionally.
    assertThat(r.hits())
        .as("at least the two exact-match queries must hit (observed %s)", r.hits())
        .isGreaterThanOrEqualTo(2);

    // "What is scalar quantization?" was never seeded and is semantically distant from the three
    // seeded answers — it must miss. So misses >= 1.
    assertThat(r.misses())
        .as("the novel query must miss (observed %s misses)", r.misses())
        .isGreaterThanOrEqualTo(1);

    // Every miss triggers the simulated LLM call exactly once.
    assertThat(r.llmCalls())
        .as(
            "LLM call count must equal miss count (1 call per miss; observed misses=%s)",
            r.misses())
        .isEqualTo(r.misses());

    // After the run, the cache must contain every seeded entry (3) plus every missed query.
    assertThat(r.entries())
        .as("entries == seeded(3) + misses(%s)", r.misses())
        .isEqualTo(3L + r.misses());
  }
}
