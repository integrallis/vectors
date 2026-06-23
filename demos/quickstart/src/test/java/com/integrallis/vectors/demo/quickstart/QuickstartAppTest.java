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
package com.integrallis.vectors.demo.quickstart;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.db.SearchResult;
import org.junit.jupiter.api.Test;

/**
 * CI gate for the quickstart demo (audit T3.10). The demo previously printed but never asserted, so
 * a regression in the HNSW search path or the document insertion API could only be caught by a
 * human reading the printed output. This test calls {@link QuickstartApp#runDemo()} and asserts the
 * golden top-3 result and the expected ranking.
 */
class QuickstartAppTest {

  @Test
  void returnsExpectedTopThreeForReddishQuery() {
    SearchResult result = QuickstartApp.runDemo();

    assertThat(result.hits()).hasSize(3);

    // The query [0.8, 0.15, 0.05, 0.05] is in the reddish/warm corner of the unit cube; the
    // top-3 must contain both warm-color docs and exclude the two blue docs.
    var ids = result.hits().stream().map(SearchResult.Hit::id).toList();
    assertThat(ids)
        .as("warm-color query must surface the warm-color docs first")
        .contains("red", "fire")
        .doesNotContain("sky", "ocean");

    // Scores must be descending.
    for (int i = 1; i < result.hits().size(); i++) {
      assertThat(result.hits().get(i).score())
          .as("scores must be sorted descending")
          .isLessThanOrEqualTo(result.hits().get(i - 1).score());
    }

    // includeText=true was set on the search; the top hit's text must come through.
    SearchResult.Hit top = result.hits().get(0);
    assertThat(top.document()).isNotNull();
    assertThat(top.document().text()).isNotBlank();
  }
}
