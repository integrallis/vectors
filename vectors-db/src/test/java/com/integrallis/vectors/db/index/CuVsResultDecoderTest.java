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
  class DistanceNegation {

    @Test
    void euclideanDistancesAreNegated() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(7, 1.0f);
      row.put(3, 0.25f);
      row.put(11, 4.0f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.EUCLIDEAN);

      // Smallest distance (0.25) becomes the largest score (-0.25), sorted descending.
      assertThat(out.ordinals()).containsExactly(3, 7, 11);
      assertThat(out.scores()).containsExactly(-0.25f, -1.0f, -4.0f);
    }

    @Test
    void cosineDistancesAreNegated() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, 0.9f);
      row.put(2, 0.1f);
      row.put(3, 0.5f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.COSINE);

      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(-0.1f, -0.5f, -0.9f);
    }

    @Test
    void dotProductIsPassedThroughAsScore() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, 0.3f);
      row.put(2, 0.9f);
      row.put(3, 0.6f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.DOT_PRODUCT);

      // Higher inner products already correspond to higher similarity — no negation.
      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(0.9f, 0.6f, 0.3f);
    }

    @Test
    void maximumInnerProductIsPassedThroughAsScore() {
      Map<Integer, Float> row = new LinkedHashMap<>();
      row.put(1, -2.0f);
      row.put(2, 3.5f);
      row.put(3, 1.1f);

      IndexSpi.SearchOutcome out =
          CuVsResultDecoder.decode(fixture(row), SimilarityFunction.MAXIMUM_INNER_PRODUCT);

      assertThat(out.ordinals()).containsExactly(2, 3, 1);
      assertThat(out.scores()).containsExactly(3.5f, 1.1f, -2.0f);
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
      assertThat(out.scores()).containsExactly(-1.5f);
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
      assertThat(out.scores()).containsExactly(-2.0f, -2.0f, -2.0f);
    }
  }
}
