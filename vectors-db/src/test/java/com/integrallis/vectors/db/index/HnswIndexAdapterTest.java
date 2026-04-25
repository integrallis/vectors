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
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link HnswIndexAdapter} — the in-memory HNSW backend wired into {@code
 * VectorCollection} through {@link IndexSpi}. Tests assert BEHAVIOR (recall against brute-force
 * ground truth, correct empty-state handling, adapter rebuild semantics) rather than shape (raw
 * array lengths or ordering), per the project's {@code tests must assert behavior, not shape}
 * guideline.
 */
@Tag("unit")
class HnswIndexAdapterTest {

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

  private static float[] randomQuery(int dim, long seed) {
    Random rng = new Random(seed);
    float[] q = new float[dim];
    for (int i = 0; i < dim; i++) {
      q[i] = rng.nextFloat() * 2f - 1f;
    }
    return q;
  }

  /**
   * Normalizes a vector to unit length in-place. Required for {@link
   * SimilarityFunction#DOT_PRODUCT} with HNSW because the normalized score {@code (1+dot)/2} can go
   * negative when {@code |dot| > 1}, and {@code NodeQueue} rejects negative scores (long-encoded
   * heap ordering depends on bit 63 = 0).
   */
  private static float[] normalize(float[] v) {
    double sq = 0.0;
    for (float x : v) {
      sq += x * x;
    }
    float norm = (float) Math.sqrt(sq);
    if (norm == 0f) {
      return v;
    }
    float[] out = new float[v.length];
    for (int i = 0; i < v.length; i++) {
      out[i] = v[i] / norm;
    }
    return out;
  }

  private static float[][] normalizeAll(float[][] vectors) {
    float[][] out = new float[vectors.length][];
    for (int i = 0; i < vectors.length; i++) {
      out[i] = normalize(vectors[i]);
    }
    return out;
  }

  private static double recall(int[] expected, int[] actual) {
    Set<Integer> expectedSet = new HashSet<>();
    for (int id : expected) {
      expectedSet.add(id);
    }
    int hits = 0;
    for (int id : actual) {
      if (expectedSet.contains(id)) {
        hits++;
      }
    }
    return (double) hits / expected.length;
  }

  @Nested
  class Construction {

    @Test
    void nonPositiveMaxConnectionsRejected() {
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new HnswIndexAdapter(0, 100))
          .withMessageContaining("maxConnections");
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new HnswIndexAdapter(-1, 100))
          .withMessageContaining("maxConnections");
    }

    @Test
    void efConstructionBelowMaxConnectionsRejected() {
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> new HnswIndexAdapter(16, 8))
          .withMessageContaining("efConstruction");
    }

    @Test
    void efConstructionEqualToMaxConnectionsAccepted() {
      // Boundary: efConstruction == maxConnections is legal.
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 16);
      assertThat(adapter.maxConnections()).isEqualTo(16);
      assertThat(adapter.efConstruction()).isEqualTo(16);
    }
  }

  @Nested
  class BuildAndSearch {

    @Test
    void buildAndSearchReturnsAccurateNeighbors() {
      // Assert BEHAVIOR (recall), not SHAPE (lengths). A constant-scoring stub would pass
      // a shape test but fail this recall assertion.
      int dim = 32;
      int n = 200;
      int k = 10;
      float[][] vectors = randomVectors(n, dim, 42L);
      float[] query = randomQuery(dim, 43L);

      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(vectors, SimilarityFunction.EUCLIDEAN);
      IndexSpi.SearchOutcome expected = reference.search(query, k, 0, 1f);

      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(vectors, SimilarityFunction.EUCLIDEAN);
      IndexSpi.SearchOutcome actual = adapter.search(query, k, 100, 1f);

      assertThat(actual.ordinals()).hasSize(k);
      assertThat(actual.scores()).hasSize(k);
      // Scores descending.
      assertThat(actual.scores()).isSortedAccordingTo((a, b) -> Float.compare(b, a));
      // Recall ≥ 0.8 on small random corpus (HNSW on 200 × 32-dim is not deterministic brute
      // force but consistently clears this bar).
      assertThat(recall(expected.ordinals(), actual.ordinals())).isGreaterThanOrEqualTo(0.8);
    }

    @Test
    void searchWorksAcrossAllSimilarityFunctions() {
      // Use unit-normalized vectors so DOT_PRODUCT (and MAXIMUM_INNER_PRODUCT) stay in the
      // non-negative score range that NodeQueue requires for its long-encoded ordering.
      //
      // Recall floor is 0.6 (not 0.8) because MAXIMUM_INNER_PRODUCT uses
      // SimilarityFunction.scaleMaxInnerProductScore which maps raw inner products through a
      // piecewise function distinct from the linear (1+x)/2 used by DOT_PRODUCT. On
      // unit-normalized random vectors the two metrics produce slightly different top-k
      // orderings for the same graph, and on small corpora (n=150) that variance is enough to
      // push MIP recall below 0.8 occasionally while the other three metrics clear 0.8 easily.
      // The point of this test is SMOKE-LEVEL coverage across all four metrics — that the
      // adapter builds and searches without error for each — not a tight recall assertion.
      // The dedicated BuildAndSearch test above enforces the stricter 0.8 floor for EUCLIDEAN.
      int dim = 16;
      int n = 150;
      int k = 5;
      float[][] vectors = normalizeAll(randomVectors(n, dim, 7L));
      float[] query = normalize(randomQuery(dim, 8L));

      for (SimilarityFunction sim : SimilarityFunction.values()) {
        FlatScanAdapter reference = new FlatScanAdapter();
        reference.build(vectors, sim);
        IndexSpi.SearchOutcome expected = reference.search(query, k, 0, 1f);

        HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
        adapter.build(vectors, sim);
        IndexSpi.SearchOutcome actual = adapter.search(query, k, 100, 1f);

        assertThat(actual.ordinals()).as("metric=%s ordinals", sim).hasSize(k);
        assertThat(recall(expected.ordinals(), actual.ordinals()))
            .as("metric=%s recall", sim)
            .isGreaterThanOrEqualTo(0.6);
      }
    }

    @Test
    void rebuildOverwritesPreviousGraph() {
      int dim = 16;
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);

      float[][] first = randomVectors(50, dim, 1L);
      adapter.build(first, SimilarityFunction.EUCLIDEAN);
      assertThat(adapter.size()).isEqualTo(50);

      float[][] second = randomVectors(120, dim, 2L);
      adapter.build(second, SimilarityFunction.EUCLIDEAN);
      assertThat(adapter.size()).isEqualTo(120);

      // Search results must reflect the SECOND dataset. We verify by checking that the returned
      // ordinals are in the range [0, 120) — the first dataset only has 50 ids.
      float[] query = randomQuery(dim, 3L);
      IndexSpi.SearchOutcome out = adapter.search(query, 10, 100, 1f);
      for (int id : out.ordinals()) {
        assertThat(id).isBetween(0, 119);
      }
      // Further: rebuild correctness — brute-force the second dataset and check recall.
      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(second, SimilarityFunction.EUCLIDEAN);
      IndexSpi.SearchOutcome expected = reference.search(query, 10, 0, 1f);
      assertThat(recall(expected.ordinals(), out.ordinals())).isGreaterThanOrEqualTo(0.7);
    }

    @Test
    void searchHonorsSearchListSizeAsEfSearchFloor() {
      // searchListSize flows into HnswSearcher.efSearch with a clamp at k. We can't directly
      // observe efSearch but we can verify: a call with searchListSize < k still returns exactly
      // k results (no throw), and a call with searchListSize > k also returns k results. Both
      // paths exercise the Math.max(searchListSize, k) clamp in the adapter.
      int dim = 16;
      float[][] vectors = randomVectors(50, dim, 1L);
      float[] query = randomQuery(dim, 2L);

      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(vectors, SimilarityFunction.EUCLIDEAN);

      IndexSpi.SearchOutcome smallBeam = adapter.search(query, 10, 1, 1f);
      IndexSpi.SearchOutcome largeBeam = adapter.search(query, 10, 200, 1f);
      assertThat(smallBeam.ordinals()).hasSize(10);
      assertThat(largeBeam.ordinals()).hasSize(10);
    }
  }

  @Nested
  class EmptyState {

    @Test
    void freshAdapterHasZeroSize() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      assertThat(adapter.size()).isZero();
      assertThat(adapter.graph()).isNull();
    }

    @Test
    void buildWithEmptyArraySizeIsZero() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(new float[0][], SimilarityFunction.EUCLIDEAN);
      assertThat(adapter.size()).isZero();
      assertThat(adapter.graph()).isNull();
    }

    @Test
    void searchOnEmptyAdapterReturnsEmptyOutcome() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(new float[0][], SimilarityFunction.EUCLIDEAN);
      IndexSpi.SearchOutcome out = adapter.search(new float[8], 10, 100, 1f);
      assertThat(out.ordinals()).isEmpty();
      assertThat(out.scores()).isEmpty();
    }

    @Test
    void searchOnNeverBuiltAdapterReturnsEmptyOutcome() {
      // The commit pipeline never calls search() on an un-built adapter, but defensive behavior
      // here prevents NPEs during edge cases (e.g. persistent bootstrap).
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      IndexSpi.SearchOutcome out = adapter.search(new float[8], 10, 100, 1f);
      assertThat(out.ordinals()).isEmpty();
      assertThat(out.scores()).isEmpty();
    }
  }

  @Nested
  class InputValidation {

    @Test
    void buildNullVectorsThrows() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      assertThatNullPointerException()
          .isThrownBy(() -> adapter.build(null, SimilarityFunction.EUCLIDEAN))
          .withMessageContaining("vectors");
    }

    @Test
    void buildNullMetricThrows() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      assertThatNullPointerException()
          .isThrownBy(() -> adapter.build(new float[1][8], null))
          .withMessageContaining("metric");
    }

    @Test
    void searchNullQueryThrows() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(randomVectors(20, 8, 1L), SimilarityFunction.EUCLIDEAN);
      assertThatNullPointerException()
          .isThrownBy(() -> adapter.search(null, 5, 100, 1f))
          .withMessageContaining("query");
    }

    @Test
    void searchNonPositiveKRejected() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(randomVectors(20, 8, 1L), SimilarityFunction.EUCLIDEAN);
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> adapter.search(new float[8], 0, 100, 1f))
          .withMessageContaining("k must be positive");
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> adapter.search(new float[8], -1, 100, 1f))
          .withMessageContaining("k must be positive");
    }

    @Test
    void searchWrongDimensionQueryThrows() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(randomVectors(20, 8, 1L), SimilarityFunction.EUCLIDEAN);
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> adapter.search(new float[9], 5, 100, 1f))
          .withMessageContaining("dimension");
    }
  }

  @Nested
  class Accessors {

    @Test
    void buildParametersArePreserved() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(8, 32);
      assertThat(adapter.maxConnections()).isEqualTo(8);
      assertThat(adapter.efConstruction()).isEqualTo(32);
    }

    @Test
    void graphAccessorReturnsCurrentGraphAfterBuild() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(randomVectors(30, 8, 1L), SimilarityFunction.EUCLIDEAN);
      assertThat(adapter.graph()).isNotNull();
      assertThat(adapter.graph().size()).isEqualTo(30);
    }

    @Test
    void graphAccessorReturnsNullBeforeBuild() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      assertThat(adapter.graph()).isNull();
    }
  }

  @Nested
  class CloseSemantics {

    @Test
    void closeIsNoOp() {
      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(randomVectors(20, 8, 1L), SimilarityFunction.EUCLIDEAN);
      adapter.close();
      // Still usable after close().
      IndexSpi.SearchOutcome out = adapter.search(new float[8], 5, 100, 1f);
      assertThat(out.ordinals()).hasSize(5);
    }
  }

  @Nested
  class ThreadSafety {

    @Test
    void concurrentSearchesProduceConsistentResults() throws Exception {
      // HnswIndex uses ThreadLocal<HnswSearcher> so each thread owns its scratch BitSet.
      // Regression guard: two threads running queries against the same adapter must not corrupt
      // each other's visited state. EUCLIDEAN is used because its similarity score 1/(1+d²) is
      // always non-negative — required by NodeQueue's long-encoded heap ordering.
      int dim = 32;
      int n = 300;
      int k = 10;
      float[][] vectors = randomVectors(n, dim, 42L);

      HnswIndexAdapter adapter = new HnswIndexAdapter(16, 100);
      adapter.build(vectors, SimilarityFunction.EUCLIDEAN);

      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(vectors, SimilarityFunction.EUCLIDEAN);

      int threads = 4;
      int iterationsPerThread = 50;
      ExecutorService pool = Executors.newFixedThreadPool(threads);
      CountDownLatch start = new CountDownLatch(1);
      CountDownLatch done = new CountDownLatch(threads);
      AtomicReference<Throwable> failure = new AtomicReference<>();

      for (int t = 0; t < threads; t++) {
        final int seed = t;
        pool.submit(
            () -> {
              try {
                start.await();
                Random rng = new Random(seed);
                for (int iter = 0; iter < iterationsPerThread; iter++) {
                  float[] q = new float[dim];
                  for (int d = 0; d < dim; d++) {
                    q[d] = rng.nextFloat() * 2f - 1f;
                  }
                  IndexSpi.SearchOutcome expected = reference.search(q, k, 0, 1f);
                  IndexSpi.SearchOutcome actual = adapter.search(q, k, 100, 1f);
                  // Per-thread recall must stay within HNSW tolerance; cross-thread interference
                  // would manifest as stale ordinals, not soft recall drift.
                  if (recall(expected.ordinals(), actual.ordinals()) < 0.6) {
                    throw new AssertionError(
                        "recall below floor on thread "
                            + seed
                            + " iter "
                            + iter
                            + " — possible cross-thread scratch corruption");
                  }
                }
              } catch (Throwable th) {
                failure.compareAndSet(null, th);
              } finally {
                done.countDown();
              }
            });
      }

      start.countDown();
      assertThat(done.await(30, TimeUnit.SECONDS)).isTrue();
      pool.shutdownNow();

      if (failure.get() != null) {
        throw new AssertionError("concurrent search failed", failure.get());
      }
    }
  }
}
