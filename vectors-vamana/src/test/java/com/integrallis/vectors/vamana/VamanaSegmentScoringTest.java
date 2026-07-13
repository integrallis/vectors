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
package com.integrallis.vectors.vamana;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Verifies the zero-copy {@link MemorySegment} scoring path in {@link VamanaSearcher} (audit vamana
 * #1). The store below reports {@code supportsSegments() == true} and serves {@link
 * #vectorSegment(int)} from an off-heap segment, while its {@link #getVector(int)} THROWS — so a
 * search that reaches this store MUST use the segment path. Without the fix (no segment branch in
 * the scorer factory, and no {@code supportsSegments()} on the vamana interface), the searcher would
 * fall back to {@code getVector} and the search would blow up.
 */
@Tag("unit")
class VamanaSegmentScoringTest {

  /** Segment-only store: vectorSegment() works, getVector() throws to force the zero-copy path. */
  private static final class SegmentOnlyVectors implements RandomAccessVectors {
    private final int n;
    private final int dim;
    private final MemorySegment seg; // n * dim floats, row-major, off-heap

    SegmentOnlyVectors(float[][] data, Arena arena) {
      this.n = data.length;
      this.dim = data[0].length;
      this.seg = arena.allocate((long) n * dim * Float.BYTES);
      for (int i = 0; i < n; i++) {
        MemorySegment.copy(
            data[i], 0, seg, ValueLayout.JAVA_FLOAT, (long) i * dim * Float.BYTES, dim);
      }
    }

    @Override
    public int size() {
      return n;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public float[] getVector(int ordinal) {
      throw new UnsupportedOperationException("segment path must be used, not getVector()");
    }

    @Override
    public boolean supportsSegments() {
      return true;
    }

    @Override
    public MemorySegment vectorSegment(int ordinal) {
      return seg.asSlice((long) ordinal * dim * Float.BYTES, (long) dim * Float.BYTES);
    }
  }

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] data = new float[n][dim];
    for (float[] row : data) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return data;
  }

  private static int[] bruteForceTopK(float[][] data, float[] query, int k) {
    float[] best = new float[k];
    int[] bestId = new int[k];
    java.util.Arrays.fill(best, Float.MAX_VALUE);
    java.util.Arrays.fill(bestId, -1);
    for (int i = 0; i < data.length; i++) {
      float d = VectorUtil.squareDistance(query, data[i]);
      if (d < best[k - 1]) {
        int pos = k - 1;
        while (pos > 0 && best[pos - 1] > d) {
          best[pos] = best[pos - 1];
          bestId[pos] = bestId[pos - 1];
          pos--;
        }
        best[pos] = d;
        bestId[pos] = i;
      }
    }
    return bestId;
  }

  @Test
  void segmentPathSearchesCorrectlyWithoutGetVector() {
    int n = 1500;
    int dim = 32;
    int k = 10;
    float[][] data = randomVectors(n, dim, 7L);

    // Build the graph from a stable float[][] store (build legitimately uses getVector()).
    VamanaGraph graph =
        VamanaGraphBuilder.create(
                32, 64, 1.2f, new InMemoryVectors(data), SimilarityFunction.EUCLIDEAN, 42L)
            .build();

    try (Arena arena = Arena.ofConfined()) {
      SegmentOnlyVectors segVectors = new SegmentOnlyVectors(data, arena);
      assertThat(segVectors.supportsSegments()).isTrue();

      // Search the SAME graph two ways: the zero-copy segment store, and a stable float[][] store.
      VamanaIndex segIndex =
          VamanaIndex.ofPrebuilt(graph, segVectors, SimilarityFunction.EUCLIDEAN);
      VamanaIndex fltIndex =
          VamanaIndex.ofPrebuilt(graph, new InMemoryVectors(data), SimilarityFunction.EUCLIDEAN);

      Random rng = new Random(1L);
      double segRecall = 0;
      double fltRecall = 0;
      int nQueries = 200;
      for (int q = 0; q < nQueries; q++) {
        float[] query = data[rng.nextInt(n)];
        Set<Integer> truth = new HashSet<>();
        for (int t : bruteForceTopK(data, query, k)) truth.add(t);

        // The segment path must run to completion without ever calling getVector() (which throws).
        SearchResult sr = segIndex.search(query, k, 64);
        SearchResult fr = fltIndex.search(query, k, 64);

        segRecall += overlap(sr, truth) / (double) k;
        fltRecall += overlap(fr, truth) / (double) k;
      }
      segRecall /= nQueries;
      fltRecall /= nQueries;

      // The zero-copy path must be as accurate as the float[][] path (identical graph, near-identical
      // kernels) and genuinely high-recall — proving it computed real distances, not garbage.
      assertThat(segRecall).as("segment-path recall").isGreaterThan(0.95);
      assertThat(segRecall)
          .as("segment path matches the float[][] path within kernel-rounding noise")
          .isCloseTo(fltRecall, org.assertj.core.data.Offset.offset(0.02));
    }
  }

  private static int overlap(SearchResult r, Set<Integer> truth) {
    int hits = 0;
    for (int i = 0; i < r.size(); i++) if (truth.contains(r.nodeId(i))) hits++;
    return hits;
  }
}
