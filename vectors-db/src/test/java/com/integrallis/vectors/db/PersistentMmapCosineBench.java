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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Cumulative #A+#B+#C measurement on the PERSISTENT (mmap) HNSW tier — the object-storage serving
 * path. Both collections are built, closed, and REOPENED so search runs through the mmap-backed
 * {@code MappedHnswIndexAdapter} + {@code MemorySegmentRandomAccessVectors} (the zero-copy segment
 * scorer). The only difference:
 *
 * <ul>
 *   <li><b>default (#A)</b> — COSINE normalized at ingest, scored as a fused DOT segment GEMV (#C)
 *   <li><b>preserveOriginalVectors</b> — verbatim vectors, scored with the per-row segment cosine
 *       (#B)
 * </ul>
 *
 * The delta is the combined win of #A (cosine-&gt;dot) and #C (fused GEMV) on the mmap tier. Tagged
 * {@code slow} so it is excluded from the default {@code test} run; execute with {@code ./gradlew
 * :vectors-db:slowTest --tests "*PersistentMmapCosineBench*"}. Not a correctness gate — it only
 * asserts both paths produce results; the numbers are printed and written to {@code
 * /tmp/persistent_bench.txt}.
 */
@Tag("slow")
class PersistentMmapCosineBench {

  private static final int DIM = 768;
  private static final int N = 10_000;
  private static final int K = 10;
  private static final int WARMUP = 80;
  private static final int MEASURED = 400;

  @Test
  void cumulativeMmapTierWin(@TempDir Path dir) throws IOException {
    Random rnd = new Random(2024L);
    List<Document> docs = new ArrayList<>(N);
    for (int i = 0; i < N; i++) {
      docs.add(Document.of("id-" + i, randomVector(rnd, DIM)));
    }
    float[][] queries = new float[WARMUP + MEASURED][];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = randomVector(rnd, DIM);
    }

    Path aDir = Files.createDirectories(dir.resolve("a")); // #A default
    Path bDir = Files.createDirectories(dir.resolve("b")); // preserveOriginalVectors
    buildAndClose(aDir, docs, false);
    buildAndClose(bDir, docs, true);

    try (VectorCollection a = open(aDir, false); // mmap segment path, fused DOT
        VectorCollection b = open(bDir, true)) { // mmap segment path, per-row cosine
      double aQps = bench(a, queries);
      double bQps = bench(b, queries);
      String report =
          String.format(
              "persistent mmap cosine bench (#A+#B+#C), N=%d dim=%d k=%d measured=%d:%n"
                  + "  preserveOriginalVectors (cosine per-row segment): %.1f qps%n"
                  + "  #A default (normalize->fused DOT segment GEMV)  : %.1f qps%n"
                  + "  speedup: %.2fx%n",
              N, DIM, K, MEASURED, bQps, aQps, aQps / bQps);
      System.out.print(report);
      Files.writeString(Paths.get("/tmp/persistent_bench.txt"), report);
      assertThat(aQps).isGreaterThan(0.0);
      assertThat(bQps).isGreaterThan(0.0);
    }
  }

  private static void buildAndClose(Path root, List<Document> docs, boolean preserve) {
    VectorCollection c = open(root, preserve);
    c.addAll(docs);
    c.commit();
    c.close();
  }

  private static VectorCollection open(Path root, boolean preserve) {
    VectorCollectionBuilder b =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .storagePath(root.toAbsolutePath());
    if (preserve) {
      b = b.preserveOriginalVectors(true);
    }
    return b.build();
  }

  private static double bench(VectorCollection c, float[][] queries) {
    for (int i = 0; i < WARMUP; i++) {
      c.search(SearchRequest.builder(queries[i], K).build());
    }
    long checksum = 0;
    long start = System.nanoTime();
    for (int i = WARMUP; i < queries.length; i++) {
      checksum += c.search(SearchRequest.builder(queries[i], K).build()).hits().size();
    }
    long ns = System.nanoTime() - start;
    if (checksum == Long.MIN_VALUE) {
      throw new IllegalStateException("unreachable");
    }
    return MEASURED / (ns / 1e9);
  }

  private static float[] randomVector(Random rnd, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = (rnd.nextFloat() * 2f - 1f) * (1f + rnd.nextFloat() * 9f);
    }
    return v;
  }
}
