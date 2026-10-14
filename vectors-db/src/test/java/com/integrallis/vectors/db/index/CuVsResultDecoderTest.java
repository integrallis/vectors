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
package com.integrallis.vectors.db.index;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.nvidia.cuvs.SearchResults;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Unit tests for {@link CuVsResultDecoder}: distance-to-similarity transform per metric, descending
 * sort, and empty-result handling. No GPU required — works against an in-memory {@link
 * SearchResults} fixture.
 */
@Tag("unit")
class CuVsResultDecoderTest {

  private static SearchResults fixture(Map<Integer, Float> row) {
    return () -> List.of(row);
  }

  @Nested
  class ScoreNormalization {

    @Test
    void squaredEuclideanDistancesBecomeNormalizedScores() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(7, 1.0f);
      row.put(3, 0.25f);
      row.put(11, 4.0f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.EUCLIDEAN);

      // Smallest squared distance becomes the largest normalized score.
      assertThat(out.ordinals()).containsExactly(3, 7, 11);
      assertThat(out.scores()).containsExactly(0.8f, 0.5f, 0.2f);
    }

    @Test
    void cosineDistancesBecomeNormalizedScores() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, 0.9f);
      row.put(2, 0.1f);
      row.put(3, 0.5f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.COSINE);

      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(0.95f, 0.75f, 0.55f);
    }

    @Test
    void dotProductsBecomeNormalizedScores() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, 0.3f);
      row.put(2, 0.9f);
      row.put(3, 0.6f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.DOT_PRODUCT);

      // The score transform preserves the ordering of inner products.
      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(0.95f, 0.8f, 0.65f);
    }

    @Test
    void maximumInnerProductsUseThePublicPiecewiseScore() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, -2.0f);
      row.put(2, 3.5f);
      row.put(3, 1.1f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.MAXIMUM_INNER_PRODUCT);

      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(4.5f, 2.1f, 1f / 3f);
    }
  }

  @ParameterizedTest
  @EnumSource(SimilarityFunction.class)
  void decodedScoresMatchCpuScoresForTheSameVectors(SimilarityFunction metric) {
    float[] query = {1f, 0f};
    float[][] vectors = {{1f, 0f}, {0f, 1f}, {-1f, 0f}};
    // Exact raw distances for the same, orthogonal, and opposite unit vectors.
    float[] raw =
        switch (metric) {
          case EUCLIDEAN -> new float[] {0f, 2f, 4f};
          case COSINE -> new float[] {0f, 1f, 2f};
          case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> new float[] {1f, 0f, -1f};
        };
    Map<Integer, Float> row = new LinkedHashMap<>();
    for (int i = vectors.length - 1; i >= 0; i--) row.put(i, raw[i]);

    IndexSpi.SearchOutcome result = CuVsResultDecoder.decode(fixture(row), metric);

    assertThat(result.ordinals()).containsExactly(0, 1, 2);
    for (int i = 0; i < vectors.length; i++) {
      assertThat(result.scores()[i]).isEqualTo(metric.compare(query, vectors[i]));
    }
  }

  @Nested
  class EdgeCases {

    @Test
    void emptyResultsReturnEmptyOutcome() {
      SearchResults empty = List::of;

      IndexSpi.SearchOutcome out = CuVsResultDecoder.decode(empty, SimilarityFunction.EUCLIDEAN);

      assertThat(out.ordinals()).isEmpty();
      assertThat(out.scores()).isEmpty();
    }

    @Test
    void singleResultRow() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(42, 1.5f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.EUCLIDEAN);

      assertThat(out.ordinals()).containsExactly(42);
      assertThat(out.scores()).containsExactly(0.4f);
    }

    @Test
    void tiesPreserveStableOutput() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, 2.0f);
      row.put(2, 2.0f);
      row.put(3, 2.0f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.EUCLIDEAN);

      assertThat(out.ordinals()).hasSize(3).containsOnly(1, 2, 3);
      assertThat(out.scores()).containsExactly(1f / 3f, 1f / 3f, 1f / 3f);
    }
  }
}
