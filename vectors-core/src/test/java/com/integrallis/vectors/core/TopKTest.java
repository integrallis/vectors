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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for the {@link TopK} selection utility. */
@Tag("unit")
class TopKTest {

  @Test
  void select_basicTop3() {
    float[] scores = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};
    assertThat(TopK.select(scores, 3)).containsExactly(1, 3, 4);
  }

  @Test
  void select_descendingOrder() {
    float[] scores = {5f, 1f, 4f, 2f, 3f};
    assertThat(TopK.select(scores, 5)).containsExactly(0, 2, 4, 3, 1);
  }

  @Test
  void select_kGreaterThanN_returnsFullSort() {
    float[] scores = {0.3f, 0.1f, 0.2f};
    assertThat(TopK.select(scores, 10)).containsExactly(0, 2, 1);
  }

  @Test
  void select_kEqualsZero_returnsEmpty() {
    assertThat(TopK.select(new float[] {1f, 2f, 3f}, 0)).isEmpty();
  }

  @Test
  void select_kNegative_returnsEmpty() {
    assertThat(TopK.select(new float[] {1f, 2f, 3f}, -5)).isEmpty();
  }

  @Test
  void select_emptyScores_returnsEmpty() {
    assertThat(TopK.select(new float[0], 5)).isEmpty();
  }

  @Test
  void select_tiesBrokenByOrdinalAscending() {
    float[] scores = {0.5f, 0.5f, 0.5f, 0.5f};
    assertThat(TopK.select(scores, 2)).containsExactly(0, 1);
    assertThat(TopK.select(scores, 4)).containsExactly(0, 1, 2, 3);
  }

  @Test
  void select_mixedTiesAndDistinct() {
    float[] scores = {0.9f, 0.5f, 0.9f, 0.3f, 0.5f};
    // Top 4: scores 0.9 (ord 0), 0.9 (ord 2), 0.5 (ord 1), 0.5 (ord 4).
    assertThat(TopK.select(scores, 4)).containsExactly(0, 2, 1, 4);
  }

  @Test
  void select_matchesFullSort_randomArray() {
    Random rnd = new Random(1234L);
    int n = 1000;
    float[] scores = new float[n];
    for (int i = 0; i < n; i++) scores[i] = rnd.nextFloat();

    int k = 10;
    int[] got = TopK.select(scores, k);

    Integer[] ref = new Integer[n];
    for (int i = 0; i < n; i++) ref[i] = i;
    Arrays.sort(
        ref,
        (a, b) -> {
          int cmp = Float.compare(scores[b], scores[a]);
          return cmp != 0 ? cmp : Integer.compare(a, b);
        });

    int[] expected = new int[k];
    for (int i = 0; i < k; i++) expected[i] = ref[i];
    assertThat(got).containsExactly(expected);
  }

  @Test
  void select_singleElement() {
    assertThat(TopK.select(new float[] {42f}, 1)).containsExactly(0);
    assertThat(TopK.select(new float[] {42f}, 5)).containsExactly(0);
  }

  @Test
  void select_negativeAndPositiveScores() {
    float[] scores = {-1f, 2f, -3f, 4f, 0f};
    assertThat(TopK.select(scores, 3)).containsExactly(3, 1, 4);
  }

  @Test
  void select_nanScoreRejected() {
    float[] scores = {0.1f, Float.NaN, 0.3f};
    assertThatThrownBy(() -> TopK.select(scores, 2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("NaN");
  }

  @Test
  void select_nullScoresRejected() {
    assertThatThrownBy(() -> TopK.select(null, 3)).isInstanceOf(NullPointerException.class);
  }

  @Test
  void select_doesNotMutateInput() {
    float[] scores = {3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f};
    float[] copy = scores.clone();
    TopK.select(scores, 3);
    assertThat(scores).containsExactly(copy);
  }

  @Test
  void select_infinityScores() {
    float[] scores = {Float.POSITIVE_INFINITY, 1f, Float.NEGATIVE_INFINITY, 2f};
    assertThat(TopK.select(scores, 3)).containsExactly(0, 3, 1);
  }
}
