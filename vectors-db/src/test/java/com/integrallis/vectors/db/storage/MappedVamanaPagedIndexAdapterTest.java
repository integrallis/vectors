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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import com.integrallis.vectors.vamana.InMemoryVectors;
import com.integrallis.vectors.vamana.VamanaGraph;
import com.integrallis.vectors.vamana.VamanaGraphBuilder;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Pins that the disk-resident {@link MappedVamanaPagedIndexAdapter} returns <b>bit-identical</b>
 * results to the heap {@link MappedVamanaIndexAdapter} over the same graph + vectors (I.4). Because
 * the paged searcher reuses the unchanged beam loop and reads neighbours in the same on-disk order,
 * ids and scores must match exactly — not merely within a recall floor.
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

  @Test
  void bitIdenticalToHeapAdapterAcrossL(@TempDir Path tmp) throws IOException {
    int n = 500;
    float[][] vectors = randomVectors(n, 7L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    MemorySegment graphSeg = MemorySegment.ofArray(VamanaGraphCodec.encode(graph));

    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);

      MappedVamanaIndexAdapter heap = new MappedVamanaIndexAdapter(graph, rav, METRIC);
      MappedVamanaPagedIndexAdapter paged =
          new MappedVamanaPagedIndexAdapter(PagedVamanaTopology.open(graphSeg), rav, METRIC);

      assertThat(paged.size()).isEqualTo(heap.size()).isEqualTo(n);

      for (int qi = 0; qi < 50; qi++) {
        float[] q = randomQuery(1000L + qi);
        for (int searchL : new int[] {64, 128, 200}) {
          SearchOutcome h = heap.search(q, 10, searchL, 1.0f);
          SearchOutcome p = paged.search(q, 10, searchL, 1.0f);
          assertThat(p.ordinals()).as("ids @L=%d q=%d", searchL, qi).containsExactly(h.ordinals());
          assertThat(p.scores()).as("scores @L=%d q=%d", searchL, qi).containsExactly(h.scores());
        }
      }
    }
  }

  @Test
  void concurrentPagedSearchesMatchHeap(@TempDir Path tmp) throws Exception {
    int n = 400;
    float[][] vectors = randomVectors(n, 11L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    MemorySegment graphSeg = MemorySegment.ofArray(VamanaGraphCodec.encode(graph));

    try (Arena arena = Arena.ofShared()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, n, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
      MappedVamanaIndexAdapter heap = new MappedVamanaIndexAdapter(graph, rav, METRIC);
      MappedVamanaPagedIndexAdapter paged =
          new MappedVamanaPagedIndexAdapter(PagedVamanaTopology.open(graphSeg), rav, METRIC);

      // Precompute heap expectations single-threaded.
      int queryCount = 64;
      int[][] expectedIds = new int[queryCount][];
      for (int qi = 0; qi < queryCount; qi++) {
        expectedIds[qi] = heap.search(randomQuery(qi), 10, 128, 1.0f).ordinals();
      }

      ExecutorService pool = Executors.newFixedThreadPool(8);
      try {
        List<Future<Boolean>> futures = new ArrayList<>();
        for (int qi = 0; qi < queryCount; qi++) {
          final int q = qi;
          Callable<Boolean> task =
              () -> {
                int[] ids = paged.search(randomQuery(q), 10, 128, 1.0f).ordinals();
                return java.util.Arrays.equals(ids, expectedIds[q]);
              };
          futures.add(pool.submit(task));
        }
        for (int qi = 0; qi < queryCount; qi++) {
          assertThat(futures.get(qi).get()).as("paged==heap for q=%d", qi).isTrue();
        }
      } finally {
        pool.shutdownNow();
      }
    }
  }

  @Test
  void buildThrowsUnsupported(@TempDir Path tmp) throws IOException {
    float[][] vectors = randomVectors(20, 3L);
    VamanaGraph graph = buildGraph(vectors);
    Path file = writeVectorsBin(tmp, vectors);
    MemorySegment graphSeg = MemorySegment.ofArray(VamanaGraphCodec.encode(graph));
    try (Arena arena = Arena.ofConfined()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(file, 20, DIM, arena);
      MemorySegmentRandomAccessVectors rav = new MemorySegmentRandomAccessVectors(mapped);
      MappedVamanaPagedIndexAdapter paged =
          new MappedVamanaPagedIndexAdapter(PagedVamanaTopology.open(graphSeg), rav, METRIC);
      assertThatThrownBy(() -> paged.build(vectors, METRIC))
          .isInstanceOf(UnsupportedOperationException.class);
    }
  }
}
