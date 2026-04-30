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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class HybridSearchTest {

  @Test
  void singleRetrieverPassesThrough() {
    Retriever r = k -> List.of(new ScoredId("a", 1.0f), new ScoredId("b", 0.5f));
    var search = new HybridSearch(new RRFFusion(), r);

    List<ScoredId> results = search.search(2);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo("a");
  }

  @Test
  void multipleRetrieversRunInParallel() {
    Retriever r1 = k -> List.of(new ScoredId("a", 1.0f));
    Retriever r2 = k -> List.of(new ScoredId("b", 1.0f));
    Retriever r3 = k -> List.of(new ScoredId("c", 1.0f));
    var search = new HybridSearch(new RRFFusion(), r1, r2, r3);

    List<ScoredId> results = search.search(3);

    assertThat(results).hasSize(3);
    assertThat(results).extracting(ScoredId::id).containsExactlyInAnyOrder("a", "b", "c");
  }

  @Test
  void emptyRetrieverProducesNoResults() {
    Retriever r1 = k -> List.of();
    Retriever r2 = k -> List.of();
    var search = new HybridSearch(new RRFFusion(), r1, r2);

    List<ScoredId> results = search.search(5);

    assertThat(results).isEmpty();
  }

  @Test
  void noRetrieversThrows() {
    assertThatThrownBy(() -> new HybridSearch(new RRFFusion()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void nullFusionThrows() {
    assertThatThrownBy(() -> new HybridSearch(null, k -> List.of()))
        .isInstanceOf(NullPointerException.class);
  }
}
