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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link SearchResult} and {@link SearchResult.Hit}. */
@Tag("unit")
class SearchResultTest {

  @Test
  void rawScoreOf_reconstructsRawCosine_fromTransformedScore() {
    var hit = hitWithScore(0.75f);
    var result = new SearchResult(List.of(hit), 1_000_000L, SimilarityFunction.COSINE);
    // COSINE transform: (1 + cos) / 2 = 0.75 -> raw cos = 0.5
    assertThat(result.rawScoreOf(hit)).isCloseTo(0.5f, within(1e-5f));
  }

  @Test
  void rawScoreOf_reconstructsRawDotProduct_fromTransformedScore() {
    var hit = hitWithScore(0.75f);
    var result = new SearchResult(List.of(hit), 1_000_000L, SimilarityFunction.DOT_PRODUCT);
    // DOT_PRODUCT transform: (1 + dot) / 2 = 0.75 -> raw dot = 0.5
    assertThat(result.rawScoreOf(hit)).isCloseTo(0.5f, within(1e-5f));
  }

  @Test
  void rawScoreOf_reconstructsSquaredDistance_fromTransformedScore() {
    var hit = hitWithScore(0.5f);
    var result = new SearchResult(List.of(hit), 1_000_000L, SimilarityFunction.EUCLIDEAN);
    // EUCLIDEAN transform: 1 / (1 + d^2) = 0.5 -> d^2 = 1
    assertThat(result.rawScoreOf(hit)).isCloseTo(1.0f, within(1e-5f));
  }

  @Test
  void rawScoreOf_reconstructsMip_fromTransformedScore() {
    var hit = hitWithScore(5.0f);
    var result =
        new SearchResult(List.of(hit), 1_000_000L, SimilarityFunction.MAXIMUM_INNER_PRODUCT);
    // MIP transform: dot + 1 = 5.0 -> raw dot = 4.0
    assertThat(result.rawScoreOf(hit)).isCloseTo(4.0f, within(1e-5f));
  }

  @Test
  void rawScoreOf_withoutMetric_throws() {
    var hit = hitWithScore(0.5f);
    var result = new SearchResult(List.of(hit), 1_000_000L); // no metric
    assertThatThrownBy(() -> result.rawScoreOf(hit))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("metric not available");
  }

  @Test
  void backwardCompatibleConstructor_preservesOldApi() {
    var hit = hitWithScore(0.5f);
    var result = new SearchResult(List.of(hit), 1_000_000L);
    assertThat(result.hits()).hasSize(1);
    assertThat(result.searchTimeNanos()).isEqualTo(1_000_000L);
    // metric is null; rawScoreOf throws
    assertThatThrownBy(() -> result.rawScoreOf(hit)).isInstanceOf(IllegalStateException.class);
  }

  private static SearchResult.Hit hitWithScore(float score) {
    return new SearchResult.Hit("id-1", score, Document.of("id-1", new float[0]));
  }
}
