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

import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Micro-bench for the fused batch COSINE GEMV on the zero-copy segment scoring path.
 *
 * <p>Builds one HNSW graph over ~10k UNNORMALIZED random 768-dim vectors with COSINE similarity
 * (the preserve-original-vectors case where cosine is NOT rewritten to DOT), serves the same
 * vectors as off-heap {@link MemorySegment} slices, and measures search QPS through {@link
 * HnswSearcher} — which routes cosine segment scoring through {@link
 * com.integrallis.vectors.core.FusedSimilarity#bulkCompareSegments} → {@code
 * VectorUtil.batchCosine} (fused, query-norm-once, query chunk loaded once per 4 rows).
 *
 * <p>Writes the fused segment cosine QPS to {@code /tmp/cosine_gemv_bench.txt}. Tagged {@code slow}
 * so it is excluded from the default unit run. The parent runs the real persistent preserve-path
 * bench (baseline: 558 qps) for the authoritative comparison.
 */
@Tag("slow")
class CosineGemvBenchTest {

  private static final int N = Integer.getInteger("cosine.bench.n", 10_000);
  private static final int DIM = Integer.getInteger("cosine.bench.dim", 768);
  private static final int NQ = Integer.getInteger("cosine.bench.nq", 500);
  private static final int K = 10;
  private static final int EF = 100;

  @Test
  void fusedSegmentCosineQps() throws Exception {
    float[][] base = HnswGraphBuilderTest.randomVectors(N, DIM, 42L);
    float[][] queries = HnswGraphBuilderTest.randomVectors(NQ, DIM, 99L);

    HnswIndex index =
        HnswIndex.builder(base, SimilarityFunction.COSINE)
            .maxConnections(16)
            .efConstruction(200)
            .seed(42L)
            .build();
    HnswGraph graph = index.graph();

    try (Arena arena = Arena.ofConfined()) {
      RandomAccessVectors seg = new BenchSegmentVectors(arena, base);
      HnswSearcher searcher = new HnswSearcher(graph, seg, SimilarityFunction.COSINE);

      // Warmup (JIT the fused segment cosine kernel).
      long checksum = 0;
      for (int w = 0; w < 3; w++) {
        for (float[] q : queries) {
          checksum += searcher.search(q, K, EF).nodeIds().length;
        }
      }

      long start = System.nanoTime();
      int iters = 3;
      for (int it = 0; it < iters; it++) {
        for (float[] q : queries) {
          checksum += searcher.search(q, K, EF).nodeIds().length;
        }
      }
      long elapsedNs = System.nanoTime() - start;

      double totalQueries = (double) iters * NQ;
      double qps = totalQueries / (elapsedNs / 1e9);
      String report =
          String.format(
              "fused segment COSINE (preserve, unnormalized): N=%d dim=%d nq=%d k=%d ef=%d -> %.1f qps%n"
                  + "(checksum=%d)%n",
              N, DIM, NQ, K, EF, qps, checksum);
      Files.writeString(Path.of("/tmp/cosine_gemv_bench.txt"), report);
      System.out.print(report);
    }
  }

  /**
   * Segment-backed {@link RandomAccessVectors}: one contiguous off-heap block, zero-copy slices.
   */
  private static final class BenchSegmentVectors implements RandomAccessVectors {
    private final MemorySegment data;
    private final int size;
    private final int dim;
    private final float[] scratch;

    BenchSegmentVectors(Arena arena, float[][] vectors) {
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
