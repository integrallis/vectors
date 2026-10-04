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
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.*;
import com.integrallis.vectors.db.storage.*;
import com.integrallis.vectors.hnsw.*;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.*;

/**
 * Fixed persistent-corpus replay plus unchanged-metric controls. Uses the same frozen runtime jars.
 */
public final class CollectionSearchPerformance {
  static void flat(String name, IndexSpi index, float[][] queries, boolean batch, int dim) {
    for (int i = 0; i < 5; i++) SearchPerformance.flatPass(index, queries, batch);
    long hash = SearchPerformance.flatPass(index, queries, batch);
    for (int round = 0; round < 7; round++) {
      long start = System.nanoTime();
      long got = SearchPerformance.flatPass(index, queries, batch);
      long ns = System.nanoTime() - start;
      if (got != hash) throw new AssertionError("unstable results: " + name);
      System.out.printf(
          Locale.ROOT, "%s,%d,%d,10,%d,%d,%d%n", name, dim, index.size(), round, ns, hash);
    }
  }

  public static void main(String[] args) throws Exception {
    Path generation = Path.of(args[0]);
    var manifest = Manifest.readFrom(generation.resolve("manifest.bin"));
    int n = Math.toIntExact(manifest.liveCount()), dim = manifest.dimension();
    if (manifest.tombstoneCount() != 0)
      throw new IllegalArgumentException("live generation required");
    System.out.println("workload,dimensions,rows,k,round,nanos,result_digest");
    try (var arena = Arena.ofConfined()) {
      var store = MemorySegmentVectors.open(generation.resolve("vectors.bin"), n, dim, arena);
      float[][] data = new float[n][dim];
      for (int i = 0; i < n; i++)
        MemorySegment.copy(store.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0, data[i], 0, dim);
      float[][] queries = new float[32][dim];
      for (int i = 0; i < queries.length; i++) {
        System.arraycopy(data[(int) ((long) i * n / queries.length)], 0, queries[i], 0, dim);
        // Deterministic perturbed source samples, not a held-out recall dataset.
        queries[i][i % dim] += .001f;
      }
      var heap = new FlatScanAdapter();
      heap.build(data, manifest.metric());
      flat("flat-real-heap", heap, queries, false, dim);
      flat("flat-real-heap-batch", heap, queries, true, dim);
      flat(
          "flat-real-mapped",
          new MappedFlatScanAdapter(store, manifest.metric()),
          queries,
          false,
          dim);
      var graph = HnswGraphCodec.decode(Files.readAllBytes(generation.resolve("graph.bin")));
      RandomAccessVectors source =
          new RandomAccessVectors() {
            public int size() {
              return n;
            }

            public int dimension() {
              return dim;
            }

            public float[] getVector(int ordinal) {
              return data[ordinal];
            }

            public boolean sharesReturnBuffer() {
              return false;
            }

            public boolean supportsSegments() {
              return true;
            }

            public MemorySegment vectorSegment(int ordinal) {
              return store.vectorSlice(ordinal);
            }
          };
      var index = HnswIndex.ofPrebuilt(graph, source, manifest.metric());
      for (int ef : new int[] {32, 128, 512}) {
        for (int warm = 0; warm < 100; warm++)
          SearchPerformance.graphPass(index, queries, ef, false);
        long hash = SearchPerformance.graphPass(index, queries, ef, false);
        for (int round = 0; round < 7; round++) {
          long start = System.nanoTime();
          long got = 0;
          for (int pass = 0; pass < 10; pass++)
            got = SearchPerformance.graphPass(index, queries, ef, false);
          long ns = System.nanoTime() - start;
          if (got != hash) throw new AssertionError("unstable graph results");
          System.out.printf(
              Locale.ROOT, "hnsw-real-%d,%d,%d,10,%d,%d,%d%n", ef, dim, n, round, ns, hash);
        }
      }
      // Control workloads: these metrics do not use prepared cosine.
      float[][] controls = SearchPerformance.rows(10000, 512, 341);
      float[][] controlQueries = SearchPerformance.rows(64, 512, 713);
      for (var metric :
          new SimilarityFunction[] {SimilarityFunction.EUCLIDEAN, SimilarityFunction.DOT_PRODUCT}) {
        heap.build(controls, metric);
        flat("flat-control-" + metric, heap, controlQueries, false, 512);
        flat("flat-control-" + metric + "-batch", heap, controlQueries, true, 512);
      }
    }
  }
}
