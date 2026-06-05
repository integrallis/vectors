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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import com.integrallis.vectors.vamana.InMemoryVectors;
import com.integrallis.vectors.vamana.SearchResult;
import com.integrallis.vectors.vamana.VamanaGraph;
import com.integrallis.vectors.vamana.VamanaGraphBuilder;
import com.integrallis.vectors.vamana.VamanaIndex;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests the disk-resident {@link MappedVamanaPagedIndexAdapter} — the sole persistent Vamana
 * searcher (I.4). It pages the graph adjacency from an mmap'd {@code graph.bin}; results must be
 * <b>bit-identical</b> to the in-memory {@link VamanaIndex} over the same vectors, since the only
 * difference is the {@link com.integrallis.vectors.vamana.VamanaTopology} implementation ({@code
 * PagedVamanaTopology} vs {@code VamanaGraph}) feeding the unchanged beam loop and the same scorer.
 * Also covers recall, validation, concurrency, build-rejection, and close semantics.
 */
@Tag("unit")
class MappedVamanaPagedIndexAdapterTest {

  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;
  private static final int DIM = 32;

  private static Path writeVectorsBin(Path tmp, float[][] vectors) throws IOException {
    Path file = tmp.resolve("vectors.bin");
    try (VectorStoreWriter writer = VectorStoreWriter.open(file, DIM, VectorEncoding.FLOAT32)) {
      for (float[] v : vectors) {
        writer.writeVector(v);
      }
    }
    return file;
  }

  private static float[][] randomVectors(int n, long seed) {
    Random rng = new Random(seed);
    float[][] out = new float[n][DIM];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < DIM; j++) {
        out[i][j] = rng.nextFloat() * 2f - 1f;
      }
    }
    return out;
  }

  private static float[] randomQuery(long seed) {
    Random rng = new Random(seed);
    float[] q = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      q[i] = rng.nextFloat() * 2f - 1f;
    }
    return q;
  }

  private static VamanaGraph buildGraph(float[][] vectors) {
    return VamanaGraphBuilder.create(32, 64, 1.2f, new InMemoryVectors(vectors), METRIC, 42L)
        .build();
  }

  private static MappedVamanaPagedIndexAdapter pagedAdapter(
      VamanaGraph graph, MemorySegmentRandomAccessVectors rav) throws IOException {
    MemorySegment graphSeg = MemorySegment.ofArray(VamanaGraphCodec.encode(graph));
    return new MappedVamanaPagedIndexAdapter(PagedVamanaTopology.open(graphSeg), rav, METRIC);
  }

  private static double recall(int[] expected, int[] actual) {
    Set<Integer> exp = new HashSet<>();
    for (int e : expected) {
      exp.add(e);
    }
    int hit = 0;
    for (int a : actual) {
      if (exp.contains(a)) {
        hit++;
      }
    }
    return (double) hit / expected.length;
  }

  @Test
  void bitIdenticalToInMemoryVamanaAcrossL(@TempDir Path tmp) throws IOException {
    int n = 500;
    float[][] vectors = randomVectors(n, 7L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

      // Reference: in-memory VamanaIndex over the SAME mmap'd vectors (heap VamanaGraph topology +
      // identical scorer), so any difference would be the paged topology alone.
      VamanaIndex reference = VamanaIndex.ofPrebuilt(graph, rav, METRIC);
      MappedVamanaPagedIndexAdapter paged = pagedAdapter(graph, rav);

      assertThat(paged.size()).isEqualTo(reference.size()).isEqualTo(n);

      for (int qi = 0; qi < 50; qi++) {
        float[] q = randomQuery(1000L + qi);
        for (int searchL : new int[] {64, 128, 200}) {
          SearchResult ref = reference.search(q, 10, searchL);
          SearchOutcome p = paged.search(q, 10, searchL, 1.0f);
          assertThat(p.ordinals()).as("ids @L=%d q=%d", searchL, qi).containsExactly(ref.nodeIds());
          assertThat(p.scores()).as("scores @L=%d q=%d", searchL, qi).containsExactly(ref.scores());
        }
      }
    }
  }

  @Test
  void returnsAccurateNeighborsVsFlatScan(@TempDir Path tmp) throws IOException {
    int n = 200;
    int k = 10;
    float[][] vectors = randomVectors(n, 11L);
    float[] query = randomQuery(12L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, DIM, arena);
      MappedVamanaPagedIndexAdapter paged =
          pagedAdapter(graph, new MemorySegmentRandomAccessVectors(mapped));

      SearchOutcome actual = paged.search(query, k, 100, 1.0f);
      assertThat(actual.ordinals()).hasSize(k);
      assertThat(actual.scores()).isSortedAccordingTo((a, b) -> Float.compare(b, a));

      FlatScanAdapter reference = new FlatScanAdapter();
      reference.build(vectors, METRIC);
      SearchOutcome expected = reference.search(query, k, 0, 1.0f);
      assertThat(recall(expected.ordinals(), actual.ordinals())).isGreaterThanOrEqualTo(0.8);
    }
  }

  @Test
  void concurrentPagedSearchesMatchReference(@TempDir Path tmp) throws Exception {
    int n = 400;
    float[][] vectors = randomVectors(n, 13L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);

    try (Arena arena = Arena.ofShared()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
      VamanaIndex reference = VamanaIndex.ofPrebuilt(graph, rav, METRIC);
      MappedVamanaPagedIndexAdapter paged = pagedAdapter(graph, rav);

      int queryCount = 64;
      int[][] expected = new int[queryCount][];
      for (int qi = 0; qi < queryCount; qi++) {
        expected[qi] = reference.search(randomQuery(qi), 10, 128).nodeIds();
      }

      ExecutorService pool = Executors.newFixedThreadPool(8);
      try {
        List<Future<Boolean>> futures = new ArrayList<>();
        for (int qi = 0; qi < queryCount; qi++) {
          final int q = qi;
          Callable<Boolean> task =
              () ->
                  Arrays.equals(
                      paged.search(randomQuery(q), 10, 128, 1.0f).ordinals(), expected[q]);
          futures.add(pool.submit(task));
        }
        for (int qi = 0; qi < queryCount; qi++) {
          assertThat(futures.get(qi).get()).as("paged==reference for q=%d", qi).isTrue();
        }
      } finally {
        pool.shutdownNow();
      }
    }
  }

  @Test
  void constructorRejectsNulls(@TempDir Path tmp) throws IOException {
    float[][] vectors = randomVectors(5, 1L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    MemorySegment graphSeg = MemorySegment.ofArray(VamanaGraphCodec.encode(graph));
    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 5, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
      var topology = PagedVamanaTopology.open(graphSeg);
      assertThatNullPointerException()
          .isThrownBy(() -> new MappedVamanaPagedIndexAdapter(null, rav, METRIC));
      assertThatNullPointerException()
          .isThrownBy(() -> new MappedVamanaPagedIndexAdapter(topology, null, METRIC));
      assertThatNullPointerException()
          .isThrownBy(() -> new MappedVamanaPagedIndexAdapter(topology, rav, null));
    }
  }

  @Test
  void searchValidatesArguments(@TempDir Path tmp) throws IOException {
    float[][] vectors = randomVectors(20, 1L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 20, DIM, arena);
      MappedVamanaPagedIndexAdapter paged =
          pagedAdapter(graph, new MemorySegmentRandomAccessVectors(mapped));
      assertThatNullPointerException()
          .isThrownBy(() -> paged.search(null, 5, 100, 1.0f))
          .withMessageContaining("query");
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> paged.search(new float[DIM], 0, 100, 1.0f))
          .withMessageContaining("k must be positive");
      assertThatExceptionOfType(IllegalArgumentException.class)
          .isThrownBy(() -> paged.search(new float[DIM + 1], 3, 100, 1.0f))
          .withMessageContaining("dimension");
    }
  }

  @Test
  void buildThrowsUnsupported(@TempDir Path tmp) throws IOException {
    float[][] vectors = randomVectors(20, 3L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 20, DIM, arena);
      MappedVamanaPagedIndexAdapter paged =
          pagedAdapter(graph, new MemorySegmentRandomAccessVectors(mapped));
      assertThatThrownBy(() -> paged.build(vectors, METRIC))
          .isInstanceOf(UnsupportedOperationException.class);
    }
  }

  @Test
  void closeIsNoOpArenaOwnsLifetime(@TempDir Path tmp) throws IOException {
    float[][] vectors = randomVectors(30, 5L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 30, DIM, arena);
      MappedVamanaPagedIndexAdapter paged =
          pagedAdapter(graph, new MemorySegmentRandomAccessVectors(mapped));
      paged.close();
      // Still usable after close() — the arena is still live.
      assertThat(paged.search(randomQuery(99L), 5, 100, 1.0f).ordinals()).hasSize(5);
    }
  }
}
