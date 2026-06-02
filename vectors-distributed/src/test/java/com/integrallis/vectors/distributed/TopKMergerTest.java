/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.distributed;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.SearchResult;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TopKMergerTest {

  @Test
  void mergeDeduplicatesByHighestScoreAndSortsDescending() {
    SearchResult merged =
        TopKMerger.merge(
            List.of(result(hit("a", 0.4f), hit("b", 0.8f)), result(hit("a", 0.9f), hit("c", 0.7f))),
            2,
            123L);

    assertThat(merged.searchTimeNanos()).isEqualTo(123L);
    assertThat(merged.hits()).extracting(SearchResult.Hit::id).containsExactly("a", "b");
    assertThat(merged.hits()).extracting(SearchResult.Hit::score).containsExactly(0.9f, 0.8f);
  }

  @Test
  void mergeWithNonPositiveKReturnsEmpty() {
    SearchResult merged = TopKMerger.merge(List.of(result(hit("a", 0.9f))), 0, 456L);

    assertThat(merged.searchTimeNanos()).isEqualTo(456L);
    assertThat(merged.hits()).isEmpty();
  }

  @Test
  void mergeWithMaxIntegerKDoesNotOverflowHeapCapacity() {
    SearchResult merged =
        TopKMerger.merge(List.of(result(hit("a", 0.9f), hit("b", 0.8f))), Integer.MAX_VALUE, 789L);

    assertThat(merged.searchTimeNanos()).isEqualTo(789L);
    assertThat(merged.hits()).extracting(SearchResult.Hit::id).containsExactly("a", "b");
  }

  private static SearchResult result(SearchResult.Hit... hits) {
    return new SearchResult(List.of(hits), 1L);
  }

  private static SearchResult.Hit hit(String id, float score) {
    return new SearchResult.Hit(id, score, Document.of(id, new float[] {score}));
  }
}
