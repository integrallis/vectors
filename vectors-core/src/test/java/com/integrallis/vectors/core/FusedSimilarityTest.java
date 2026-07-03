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

import org.junit.jupiter.api.Test;

class FusedSimilarityTest {

  private static final float[] QUERY = {1f, 0f};
  private static final float[][] ROWS = {{1f, 0f}, {0f, 1f}, {-1f, 0f}};

  @Test
  void everySimilarityMatchesItsScalarComparison() {
    for (SimilarityFunction similarity : SimilarityFunction.values()) {
      float[] scratch = new float[ROWS.length];
      float[] scores = new float[ROWS.length];

      FusedSimilarity.bulkCompare(similarity, QUERY, ROWS, scratch, scores, ROWS.length);

      for (int i = 0; i < ROWS.length; i++) {
        assertThat(scores[i]).isCloseTo(similarity.compare(QUERY, ROWS[i]), within(1e-6f));
      }
    }
  }

  @Test
  void zeroCountIsAllowed() {
    FusedSimilarity.bulkCompare(
        SimilarityFunction.EUCLIDEAN, QUERY, ROWS, new float[0], new float[0], 0);
  }

  @Test
  void validatesEveryInputBoundary() {
    assertThatThrownBy(
            () -> FusedSimilarity.bulkCompare(null, QUERY, ROWS, new float[3], new float[3], 3))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, null, ROWS, new float[3], new float[3], 3))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, null, new float[3], new float[3], 3))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, null, new float[3], 3))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, new float[3], null, 3))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, new float[3], new float[3], -1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, new float[3], new float[3], 4))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, new float[2], new float[3], 3))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                FusedSimilarity.bulkCompare(
                    SimilarityFunction.COSINE, QUERY, ROWS, new float[3], new float[2], 3))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static org.assertj.core.data.Offset<Float> within(float value) {
    return org.assertj.core.data.Offset.offset(value);
  }
}
