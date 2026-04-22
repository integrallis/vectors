package com.integrallis.vectors.db.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Behavioural tests for {@link IndexSpi#searchBatch}: the default virtual-thread fan-out (exercised
 * through any backend that does not override it) and {@link FlatScanAdapter}'s transposed-scan
 * override must both produce results identical to running {@link IndexSpi#search} per query.
 */
@Tag("unit")
class IndexSpiSearchBatchTest {

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] out = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        out[i][j] = rng.nextFloat() * 2f - 1f;
      }
    }
    return out;
  }

  @Test
  void flatScanBatchEqualsSequentialSearch() {
    float[][] vectors = randomVectors(250, 16, 101L);
    float[][] queries = randomVectors(12, 16, 103L);
    FlatScanAdapter adapter = new FlatScanAdapter();
    adapter.build(vectors, SimilarityFunction.EUCLIDEAN);

    SearchOutcome[] batch = adapter.searchBatch(queries, 10, 0, 1.0f);
    assertThat(batch).hasSize(queries.length);
    for (int i = 0; i < queries.length; i++) {
      SearchOutcome sequential = adapter.search(queries[i], 10, 0, 1.0f);
      assertThat(batch[i].ordinals()).containsExactly(sequential.ordinals());
      assertThat(batch[i].scores()).containsExactly(sequential.scores());
    }
  }

  @Test
  void flatScanBatchEqualsSequentialSearchAcrossMetrics() {
    float[][] vectors = randomVectors(80, 8, 109L);
    float[][] queries = randomVectors(5, 8, 113L);
    for (SimilarityFunction metric :
        new SimilarityFunction[] {SimilarityFunction.EUCLIDEAN, SimilarityFunction.DOT_PRODUCT}) {
      FlatScanAdapter adapter = new FlatScanAdapter();
      adapter.build(vectors, metric);
      SearchOutcome[] batch = adapter.searchBatch(queries, 6, 0, 1.0f);
      for (int i = 0; i < queries.length; i++) {
        SearchOutcome seq = adapter.search(queries[i], 6, 0, 1.0f);
        assertThat(batch[i].ordinals())
            .as("metric %s, query %d", metric, i)
            .containsExactly(seq.ordinals());
        assertThat(batch[i].scores()).containsExactly(seq.scores());
      }
    }
  }

  @Test
  void singleQueryBatchMatchesScalarSearch() {
    float[][] vectors = randomVectors(50, 4, 127L);
    FlatScanAdapter adapter = new FlatScanAdapter();
    adapter.build(vectors, SimilarityFunction.EUCLIDEAN);
    float[] q = randomVectors(1, 4, 131L)[0];

    SearchOutcome[] batch = adapter.searchBatch(new float[][] {q}, 5, 0, 1.0f);
    SearchOutcome seq = adapter.search(q, 5, 0, 1.0f);
    assertThat(batch).hasSize(1);
    assertThat(batch[0].ordinals()).containsExactly(seq.ordinals());
    assertThat(batch[0].scores()).containsExactly(seq.scores());
  }

  @Test
  void emptyIndexReturnsEmptyOutcomesForEachQuery() {
    FlatScanAdapter adapter = new FlatScanAdapter();
    adapter.build(new float[0][], SimilarityFunction.EUCLIDEAN);
    SearchOutcome[] batch =
        adapter.searchBatch(new float[][] {new float[8], new float[8]}, 5, 0, 1.0f);
    assertThat(batch).hasSize(2);
    for (SearchOutcome o : batch) {
      assertThat(o.ordinals()).isEmpty();
      assertThat(o.scores()).isEmpty();
    }
  }

  @Test
  void rejectsEmptyQueryArray() {
    FlatScanAdapter adapter = new FlatScanAdapter();
    adapter.build(randomVectors(10, 4, 137L), SimilarityFunction.EUCLIDEAN);
    assertThatExceptionOfType(IllegalArgumentException.class)
        .isThrownBy(() -> adapter.searchBatch(new float[0][], 3, 0, 1.0f));
  }

  @Test
  void rejectsDimensionMismatchInAnyQuery() {
    FlatScanAdapter adapter = new FlatScanAdapter();
    adapter.build(randomVectors(10, 4, 139L), SimilarityFunction.EUCLIDEAN);
    float[][] queries = {new float[4], new float[3]};
    assertThatExceptionOfType(IllegalArgumentException.class)
        .isThrownBy(() -> adapter.searchBatch(queries, 3, 0, 1.0f));
  }

  /**
   * Exercises the {@link IndexSpi} default implementation by invoking {@code searchBatch} through a
   * wrapper that hides {@link FlatScanAdapter}'s override, forcing the virtual-thread fan-out path.
   */
  @Test
  void defaultFanOutEqualsSequentialSearch() {
    float[][] vectors = randomVectors(120, 6, 149L);
    float[][] queries = randomVectors(8, 6, 151L);
    FlatScanAdapter delegate = new FlatScanAdapter();
    delegate.build(vectors, SimilarityFunction.EUCLIDEAN);

    // Thin IndexSpi wrapper that inherits the default searchBatch from IndexSpi.
    IndexSpi defaultBatchSpi =
        new IndexSpi() {
          @Override
          public void build(float[][] v, SimilarityFunction m) {
            delegate.build(v, m);
          }

          @Override
          public SearchOutcome search(float[] q, int k, int ls, float of) {
            return delegate.search(q, k, ls, of);
          }

          @Override
          public int size() {
            return delegate.size();
          }
        };

    SearchOutcome[] batch = defaultBatchSpi.searchBatch(queries, 7, 0, 1.0f);
    assertThat(batch).hasSize(queries.length);
    for (int i = 0; i < queries.length; i++) {
      SearchOutcome seq = delegate.search(queries[i], 7, 0, 1.0f);
      assertThat(batch[i].ordinals()).containsExactly(seq.ordinals());
      assertThat(batch[i].scores()).containsExactly(seq.scores());
    }
  }
}
