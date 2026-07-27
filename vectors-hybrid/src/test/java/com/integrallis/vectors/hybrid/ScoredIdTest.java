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
package com.integrallis.vectors.hybrid;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ScoredIdTest {

  @Test
  void rejectsNaNScore() {
    // Regression: a NaN score silently poisons min/max/range in fusion and sorts as the largest
    // value under Float.compare, wedging garbage at the top of the fused ranking. It must be
    // rejected at the source rather than flow into WeightedFusion/RRFFusion.
    assertThatIllegalArgumentException()
        .isThrownBy(() -> new ScoredId("doc-1", Float.NaN))
        .withMessageContaining("doc-1");
  }

  @Test
  void acceptsFiniteAndInfiniteScores() {
    assertThat(new ScoredId("a", 0.9f).score()).isEqualTo(0.9f);
    assertThat(new ScoredId("b", 0f).score()).isZero();
    // Infinities are ordered by Float.compare (unlike NaN) so they don't poison ranking; allowed.
    assertThat(new ScoredId("c", Float.POSITIVE_INFINITY).score())
        .isEqualTo(Float.POSITIVE_INFINITY);
  }

  @Test
  void rejectsNullId() {
    assertThatNullPointerException().isThrownBy(() -> new ScoredId(null, 1.0f));
  }

  @Test
  void aNaNFromAnyRetrieverCannotReachFusion() {
    // End-to-end guard: constructing the retriever result list itself fails, so no fusion strategy
    // can ever be handed a NaN-scored hit.
    assertThatIllegalArgumentException()
        .isThrownBy(() -> List.of(new ScoredId("ok", 0.5f), new ScoredId("bad", Float.NaN)));
  }
}
