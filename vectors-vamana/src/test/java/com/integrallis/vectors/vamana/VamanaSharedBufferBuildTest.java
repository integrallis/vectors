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
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Regression for the builder shared-buffer aliasing bug (audit ivf/vamana #14). {@link
 * VamanaIndex#builder(RandomAccessVectors, SimilarityFunction)} is public API and {@link
 * RandomAccessVectors} explicitly permits an implementation whose {@link
 * RandomAccessVectors#getVector(int)} returns a single scratch buffer overwritten on every call
 * ({@link RandomAccessVectors#sharesReturnBuffer()} {@code == true}, e.g. an mmap-backed store).
 *
 * <p>The builders held {@code query = getVector(node)} across the beam search (which calls {@code
 * getVector} many times) and across a neighbor compare — so on a shared-buffer store {@code query}
 * silently aliased whatever vector was fetched last, corrupting every candidate score and producing
 * a garbage graph. These tests build from a shared-buffer store and assert the result matches the
 * stable-store build; they fail on the pre-fix builders.
 */
class VamanaSharedBufferBuildTest {

  /**
   * A shared-buffer {@link RandomAccessVectors}: every {@link #getVector(int)} returns the SAME
   * array, overwritten with the requested row. Faithfully models an mmap/scratch-row store and is
   * the minimal reproducer for the aliasing bug.
   */
  private static final class SharedBufferVectors implements RandomAccessVectors {
    private final float[][] data;
    private final float[] shared;

    SharedBufferVectors(float[][] data) {
      this.data = data;
      this.shared = new float[data[0].length];
    }

    @Override
    public int size() {
      return data.length;
    }

    @Override
    public int dimension() {
      return data[0].length;
    }

    @Override
    public float[] getVector(int ordinal) {
      System.arraycopy(data[ordinal], 0, shared, 0, shared.length);
      return shared; // same array every call — retaining it across a getVector() aliases it
    }

    // sharesReturnBuffer() intentionally left at its default of true.
  }

  /**
   * The correctness reference: returns a FRESH copy per call (so retaining it across calls is
   * always safe — no aliasing is possible) yet reports {@code sharesReturnBuffer() == true}. This
   * forces the builder/searcher down the exact same scalar scoring path as {@link
   * SharedBufferVectors}, so a graph diff between the two isolates the aliasing bug alone —
   * unclouded by the float-rounding difference between the scalar and bulk-SIMD scorers that a
   * stable {@link InMemoryVectors} (which reports {@code false}) would introduce.
   */
  private static final class FreshCopyButClaimsSharedVectors implements RandomAccessVectors {
    private final float[][] data;

    FreshCopyButClaimsSharedVectors(float[][] data) {
      this.data = data;
    }

    @Override
    public int size() {
      return data.length;
    }

    @Override
    public int dimension() {
      return data[0].length;
    }

    @Override
    public float[] getVector(int ordinal) {
      return data[ordinal].clone(); // fresh array every call — never aliased
    }

    // sharesReturnBuffer() left at its default of true, matching SharedBufferVectors.
  }

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] data = new float[n][dim];
    for (float[] row : data) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return data;
  }

  private static int[] neighborIds(NeighborArray na) {
    int[] ids = new int[na.size()];
    for (int i = 0; i < ids.length; i++) ids[i] = na.node(i);
    return ids;
  }

  @Test
  @Tag("unit")
  void serialBuildFromSharedBufferMatchesStableStore() {
    int n = 200;
    int dim = 16;
    long seed = 123L;
    float[][] data = randomVectors(n, dim, 7L);

    // Reference build: fresh-copy store on the same (scalar) scoring path, so the only variable
    // versus the shared-buffer build is whether the returned array gets aliased.
    VamanaGraph reference =
        VamanaGraphBuilder.create(
                32,
                64,
                1.2f,
                new FreshCopyButClaimsSharedVectors(data),
                SimilarityFunction.EUCLIDEAN,
                seed)
            .build();
    VamanaGraph shared =
        VamanaGraphBuilder.create(
                32, 64, 1.2f, new SharedBufferVectors(data), SimilarityFunction.EUCLIDEAN, seed)
            .build();

    // Serial construction is deterministic for a fixed seed, so a correctly-copied shared-buffer
    // build must be byte-for-byte identical to the non-aliasing reference build (both take the
    // scalar scoring path, so the only variable is whether getVector's array gets aliased).
    assertThat(shared.medoid()).isEqualTo(reference.medoid());
    assertThat(shared.size()).isEqualTo(reference.size());
    for (int node = 0; node < n; node++) {
      assertThat(neighborIds(shared.getNeighbors(node)))
          .as("neighbors of node %d must match the non-aliasing reference build", node)
          .containsExactly(neighborIds(reference.getNeighbors(node)));
    }
  }

  @Test
  @Tag("unit")
  void concurrentBuildFromSharedBufferHasHighSelfRecall() {
    int n = 300;
    int dim = 16;
    long seed = 99L;
    float[][] data = randomVectors(n, dim, 21L);

    VamanaGraph shared =
        ConcurrentVamanaGraphBuilder.create(
                32, 96, 1.2f, new SharedBufferVectors(data), SimilarityFunction.EUCLIDEAN, seed)
            .build(4);

    // Isolate build correctness from any search-side buffer handling: search the shared-buffer
    // graph through a stable-array store. A corrupted build tanks self-recall (each point should
    // trivially find itself); a correct build finds it for essentially every point.
    VamanaIndex index =
        VamanaIndex.ofPrebuilt(shared, new InMemoryVectors(data), SimilarityFunction.EUCLIDEAN);
    int hits = 0;
    for (int i = 0; i < n; i++) {
      SearchResult r = index.search(data[i], 1, 96);
      if (r.size() > 0 && r.nodeId(0) == i) hits++;
    }
    assertThat((double) hits / n)
        .as("self-recall@1 of the concurrent shared-buffer build")
        .isGreaterThanOrEqualTo(0.95);
  }
}
