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
package com.integrallis.vectors.demo.rerank;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * CI gate for the SIMD rescore demo (audit T3.10). The demo's value proposition is "a precise in-VM
 * rescore changes the ranking versus a noisy remote score" — this test pins that contract (rescore
 * is not a no-op) and that the top-K output is well-formed.
 */
class RerankAppTest {

  @Test
  void rescoreReturnsWellFormedTopKAndDiffersFromNoisyRanking() {
    RerankApp.RescoreResult r = RerankApp.runRescore(42L, 200, 10);

    assertThat(r.topIds()).hasSize(10);
    assertThat(r.topScores()).hasSize(10);

    // Scores must be sorted descending.
    for (int i = 1; i < r.topScores().length; i++) {
      assertThat(r.topScores()[i])
          .as("top-k scores must be descending")
          .isLessThanOrEqualTo(r.topScores()[i - 1]);
    }

    // Ids must be unique (no duplicate ordinals in the rescored top-K).
    assertThat(r.topIds()).doesNotHaveDuplicates();

    // The rescore must actually differ from the noisy pre-rescore ranking at least somewhere —
    // otherwise the demo's value proposition ("rescore changes the ranking") is unsubstantiated.
    assertThat(r.swapsVsNoisy())
        .as("rescore must produce at least one ranking swap vs the noisy input")
        .isPositive();
  }
}
