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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswGraphBuilder;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
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
import org.junit.jupiter.api.io.TempDir;

/**
 * Unit tests for {@link MappedHnswIndexAdapter} — the read-only HNSW SPI backed by a pre-built
 * {@link HnswGraph} plus a {@link MemorySegmentRandomAccessVectors} view of an mmap'd {@code
 * vectors.bin}. Tests assert BEHAVIOR (recall against brute-force ground truth, correct delegation
 * to {@code HnswSearcher}, thread-safe read), never shape alone.
 */
@Tag("unit")
class MappedHnswIndexAdapterTest {

  private static Path writeVectorsBin(Path tmp, float[][] vectors, int dim) throws IOException {
    Path file = tmp.resolve("vectors.bin");
    try (VectorStoreWriter writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
      for (float[] v : vectors) {
        writer.writeVector(v);
      }
    }
    return file;
  }

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

  private static HnswGraph buildGraph(float[][] vectors, SimilarityFunction sim) {
    // Build the graph against a fresh InMemoryVectors (NOT MemorySegmentRandomAccessVectors —
    // the shared-scratch contract forbids using that for the build path).
    return HnswGraphBuilder.create(16, 100, new InMemoryVectors(vectors), sim, 42L).build();
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
    void nullGraphThrows(@TempDir Path tmp) throws IOException {
      int dim = 8;
      Path file = writeVectorsBin(tmp, randomVectors(5, dim, 1L), dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 5, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        assertThatNullPointerException()
            .isThrownBy(() -> new MappedHnswIndexAdapter(null, rav, SimilarityFunction.EUCLIDEAN))
            .withMessageContaining("graph");
      }
    }

    @Test
    void nullVectorsThrows() {
      int dim = 8;
      float[][] vectors = randomVectors(5, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      assertThatNullPointerException()
          .isThrownBy(() -> new MappedHnswIndexAdapter(graph, null, SimilarityFunction.EUCLIDEAN))
          .withMessageContaining("vectors");
    }

    @Test
    void nullMetricThrows(@TempDir Path tmp) throws IOException {
      int dim = 8;
      float[][] vectors = randomVectors(5, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, vectors.length, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        assertThatNullPointerException()
            .isThrownBy(() -> new MappedHnswIndexAdapter(graph, rav, null))
            .withMessageContaining("metric");
      }
    }
  }

  @Nested
  class Search {

    @Test
    void constructedFromPreBuiltGraphReturnsAccurateNeighbors(@TempDir Path tmp)
        throws IOException {
      // Build the graph on an in-memory copy, persist the same vectors to vectors.bin, mmap them,
      // and confirm search via the mmap-backed adapter matches brute force closely.
      int dim = 32;
      int n = 200;
      int k = 10;
      float[][] vectors = randomVectors(n, dim, 11L);
      float[] query = randomQuery(dim, 12L);

      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(graph, rav, SimilarityFunction.EUCLIDEAN);

        assertThat(adapter.size()).isEqualTo(n);

        IndexSpi.SearchOutcome actual = adapter.search(query, k, 100, 1f);
        assertThat(actual.ordinals()).hasSize(k);
        assertThat(actual.scores()).hasSize(k);
        // Scores descending.
        assertThat(actual.scores()).isSortedAccordingTo((a, b) -> Float.compare(b, a));

        FlatScanAdapter reference = new FlatScanAdapter();
        reference.build(vectors, SimilarityFunction.EUCLIDEAN);
        IndexSpi.SearchOutcome expected = reference.search(query, k, 0, 1f);
        // Recall ≥ 0.8 matches the HnswIndexAdapter test's floor for in-memory HNSW. Routing
        // scoring through a MemorySegment-backed RandomAccessVectors should not degrade recall.
        assertThat(recall(expected.ordinals(), actual.ordinals())).isGreaterThanOrEqualTo(0.8);
      }
    }

    @Test
    void searchMatchesIndependentlyBuiltSearcher(@TempDir Path tmp) throws IOException {
      // Stronger equivalence test: a second MappedHnswIndexAdapter built from the same graph and
      // the same vectors must return IDENTICAL ordinals for the same query.  This is the
      // "delegation to HnswSearcher" check — if the adapter held any hidden state that leaked
      // across instances, the two results would diverge.
      int dim = 16;
      int n = 100;
      int k = 5;
      float[][] vectors = randomVectors(n, dim, 21L);
      float[] query = randomQuery(dim, 22L);

      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter a1 =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);
        MappedHnswIndexAdapter a2 =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);

        IndexSpi.SearchOutcome r1 = a1.search(query, k, 100, 1f);
        IndexSpi.SearchOutcome r2 = a2.search(query, k, 100, 1f);
        assertThat(r1.ordinals()).containsExactly(r2.ordinals());
        assertThat(r1.scores()).containsExactly(r2.scores());
      }
    }

    @Test
    void searchNullQueryThrows(@TempDir Path tmp) throws IOException {
      int dim = 8;
      int n = 5;
      float[][] vectors = randomVectors(n, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);
        assertThatNullPointerException()
            .isThrownBy(() -> adapter.search(null, 5, 100, 1f))
            .withMessageContaining("query");
      }
    }

    @Test
    void searchNonPositiveKRejected(@TempDir Path tmp) throws IOException {
      int dim = 8;
      int n = 5;
      float[][] vectors = randomVectors(n, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> adapter.search(new float[dim], 0, 100, 1f))
            .withMessageContaining("k must be positive");
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> adapter.search(new float[dim], -3, 100, 1f))
            .withMessageContaining("k must be positive");
      }
    }

    @Test
    void searchWrongDimensionQueryThrows(@TempDir Path tmp) throws IOException {
      int dim = 8;
      int n = 5;
      float[][] vectors = randomVectors(n, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);
        assertThatExceptionOfType(IllegalArgumentException.class)
            .isThrownBy(() -> adapter.search(new float[dim + 1], 3, 100, 1f))
            .withMessageContaining("dimension");
      }
    }
  }

  @Nested
  class BuildPath {

    @Test
    void buildAlwaysThrowsUnsupportedOperation(@TempDir Path tmp) throws IOException {
      // This is the critical invariant guarding the shared-scratch contract in
      // MemorySegmentRandomAccessVectors: HnswGraphBuilder.insert() would violate it by holding
      // the "query vector" reference across subsequent getVector calls for candidate neighbors.
      // The adapter's only legal construction path is from a pre-built graph, so build() must
      // be permanently inaccessible.
      int dim = 8;
      int n = 5;
      float[][] vectors = randomVectors(n, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);
        assertThatExceptionOfType(UnsupportedOperationException.class)
            .isThrownBy(() -> adapter.build(vectors, SimilarityFunction.EUCLIDEAN))
            .withMessageContaining("read-only");
      }
    }
  }

  @Nested
  class CloseSemantics {

    @Test
    void closeIsNoOpArenaOwnsLifetime(@TempDir Path tmp) throws IOException {
      // The adapter's close() must NOT close the arena; the caller owns the arena lifetime.
      // Subsequent searches inside the same arena scope must still succeed.
      int dim = 16;
      int n = 30;
      float[][] vectors = randomVectors(n, dim, 1L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);

        adapter.close();
        // Still usable after close() — the arena is still live.
        IndexSpi.SearchOutcome out = adapter.search(randomQuery(dim, 99L), 5, 100, 1f);
        assertThat(out.ordinals()).hasSize(5);
      }
    }
  }

  @Nested
  class ThreadSafety {

    @Test
    void concurrentSearchesNeverCorruptResults(@TempDir Path tmp) throws Exception {
      // Regression guard: the underlying HnswIndex uses ThreadLocal<HnswSearcher> and the
      // MemorySegmentRandomAccessVectors uses ThreadLocal<float[]>. Both must compose correctly
      // under concurrent load — cross-thread scratch contamination would appear as recall drift.
      int dim = 32;
      int n = 200;
      int k = 10;
      float[][] vectors = randomVectors(n, dim, 42L);
      HnswGraph graph = buildGraph(vectors, SimilarityFunction.EUCLIDEAN);
      Path file = writeVectorsBin(tmp, vectors, dim);

      try (Arena arena = Arena.ofShared()) {
        MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, dim, arena);
        MappedHnswIndexAdapter adapter =
            new MappedHnswIndexAdapter(
                graph, new MemorySegmentRandomAccessVectors(mapped), SimilarityFunction.EUCLIDEAN);

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
                    // Floor matches constructedFromPreBuiltGraphReturnsAccurateNeighbors' 0.8.
                    // Cross-thread scratch corruption would drop recall well below this floor
                    // (corrupted query -> wrong neighborhood), so 0.8 stays comfortably above
                    // normal variance while still catching the class of bug this test guards.
                    if (recall(expected.ordinals(), actual.ordinals()) < 0.8) {
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
}
