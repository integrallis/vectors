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
package com.integrallis.vectors.hnsw;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Proves the zero-copy segment scoring path in {@link HnswSearcher} preserves recall.
 *
 * <p>The same HNSW graph (built once over N random vectors) is searched two ways:
 *
 * <ul>
 *   <li>the {@code float[]} path — {@link InMemoryVectors} ({@code supportsSegments()==false})
 *   <li>the zero-copy path — {@link SegmentVectors} test double ({@code supportsSegments()==true})
 *       serving the SAME vectors as off-heap {@code MemorySegment} slices
 * </ul>
 *
 * For a batch of queries, the top-k node ids from both paths must be effectively identical. This is
 * the regression guard: if the segment kernels or the query-upload wiring diverged numerically, the
 * rankings would drift and this test would fail.
 */
@Tag("unit")
class HnswSegmentScoringRecallTest {

  private static final int N = 3000;
  private static final int DIM = 64;
  private static final int K = 10;
  private static final int EF = 100;

  @ParameterizedTest
  @EnumSource(SimilarityFunction.class)
  void segmentPath_topK_matchesFloatArrayPath(SimilarityFunction sim) {
    float[][] vectors = HnswGraphBuilderTest.randomVectors(N, DIM, 42L);
    float[][] queries = HnswGraphBuilderTest.randomVectors(50, DIM, 99L);

    // Build ONE graph. Both searchers share it — only the vector-access mechanism differs.
    HnswIndex index =
        HnswIndex.builder(vectors, sim).maxConnections(16).efConstruction(200).seed(42L).build();
    HnswGraph graph = index.graph();

    try (Arena arena = Arena.ofConfined()) {
      RandomAccessVectors heap = new InMemoryVectors(vectors);
      RandomAccessVectors seg = new SegmentVectors(arena, vectors);
      assertThat(heap.supportsSegments()).isFalse();
      assertThat(seg.supportsSegments()).isTrue();

      // Fresh searcher per access mechanism, same graph + same similarity function.
      HnswSearcher heapSearcher = new HnswSearcher(graph, heap, sim);
      HnswSearcher segSearcher = new HnswSearcher(graph, seg, sim);

      int totalOverlap = 0;
      int totalExpected = 0;
      for (float[] query : queries) {
        SearchResult heapResult = heapSearcher.search(query, K, EF);
        SearchResult segResult = segSearcher.search(query, K, EF);

        int[] heapIds = heapResult.nodeIds();
        int[] segIds = segResult.nodeIds();

        // Overlap of the two top-k id sets (order-insensitive; fp tie-breaking may reorder equal
        // scores, but the SET must be essentially identical).
        int overlap = overlapCount(heapIds, segIds);
        totalOverlap += overlap;
        totalExpected += heapIds.length;

        // Per-query: at least 0.99 overlap (allows a single boundary tie difference on rare ties).
        assertThat((double) overlap / heapIds.length)
            .as("%s per-query top-%d overlap", sim, K)
            .isGreaterThanOrEqualTo(0.99);
      }

      double avgOverlap = (double) totalOverlap / totalExpected;
      assertThat(avgOverlap)
          .as("%s aggregate top-%d overlap (segment vs float[] path)", sim, K)
          .isGreaterThanOrEqualTo(0.999);
    }
  }

  private static int overlapCount(int[] a, int[] b) {
    int count = 0;
    for (int x : a) {
      for (int y : b) {
        if (x == y) {
          count++;
          break;
        }
      }
    }
    return count;
  }

  /**
   * Test double: a segment-backed {@link RandomAccessVectors} that stores every vector contiguously
   * in one off-heap {@link MemorySegment} and hands out zero-copy slices via {@link
   * #vectorSegment(int)}. Mirrors what {@code MemorySegmentRandomAccessVectors} does over an mmap,
   * without pulling the vectors-db dependency into this module.
   */
  private static final class SegmentVectors implements RandomAccessVectors {
    private final MemorySegment data;
    private final int size;
    private final int dim;
    // Scratch for getVector() — exercises the (unused-on-segment-path) copy contract for
    // completeness.
    private final float[] scratch;

    SegmentVectors(Arena arena, float[][] vectors) {
      this.size = vectors.length;
      this.dim = vectors[0].length;
      this.scratch = new float[dim];
      this.data = arena.allocate((long) size * dim * Float.BYTES);
      for (int i = 0; i < size; i++) {
        MemorySegment.copy(
            vectors[i], 0, data, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
      }
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public float[] getVector(int ordinal) {
      MemorySegment.copy(
          data, ValueLayout.JAVA_FLOAT, (long) ordinal * dim * Float.BYTES, scratch, 0, dim);
      return scratch;
    }

    @Override
    public boolean sharesReturnBuffer() {
      return true;
    }

    @Override
    public boolean supportsSegments() {
      return true;
    }

    @Override
    public MemorySegment vectorSegment(int ordinal) {
      return data.asSlice((long) ordinal * dim * Float.BYTES, (long) dim * Float.BYTES);
    }
  }
}
