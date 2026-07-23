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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import com.integrallis.vectors.hnsw.ConcurrentHnswGraphBuilder;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.hnsw.RandomAccessVectorDataset;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizer;
import com.integrallis.vectors.quantization.VectorDataset;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class MappedBuildVectorsTest {

  @Test
  void mmapBuildMatchesInRamBuildAndTrainsCodes(@TempDir Path tmp) throws Exception {
    int n = 1500;
    int dim = 64;
    int k = 10;
    int nq = 20;
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    float[][] vecs = random(n, dim, 1L);
    float[][] queries = random(nq, dim, 2L);

    // In-RAM build (single-thread => deterministic).
    HnswGraph gRam =
        ConcurrentHnswGraphBuilder.create(16, 200, new InMemoryVectors(vecs), sim, 42L).build(1);

    // Write vectors.bin and build the SAME graph from the mmap — bounded-RAM path.
    Path vb = tmp.resolve("vectors.bin");
    Files.write(vb, encodeVectorsBin(vecs, dim));

    // Shared arena: the concurrent builder scores from worker threads, so the mmap segment must be
    // accessible off the owning thread (a confined arena throws WrongThreadException).
    try (Arena arena = Arena.ofShared()) {
      MemorySegmentVectors mapped = MemorySegmentVectors.open(vb, n, dim, arena);
      MappedBuildVectors mbv = new MappedBuildVectors(mapped);
      assertThat(mbv.size()).isEqualTo(n);
      assertThat(mbv.dimension()).isEqualTo(dim);
      assertThat(mbv.sharesReturnBuffer()).isFalse();
      assertThat(mbv.supportsSegments()).isTrue();

      HnswGraph gMmap = ConcurrentHnswGraphBuilder.create(16, 200, mbv, sim, 42L).build(1);

      // Deterministic build over identical vectors/params/seed => identical search results.
      MappedHnswIndexAdapter aRam =
          new MappedHnswIndexAdapter(gRam, new InMemoryVectors(vecs), sim);
      MappedHnswIndexAdapter aMmap =
          new MappedHnswIndexAdapter(gMmap, new InMemoryVectors(vecs), sim);
      for (float[] q : queries) {
        SearchOutcome r1 = aRam.search(q, k, 100, 1.0f);
        SearchOutcome r2 = aMmap.search(q, k, 100, 1.0f);
        assertThat(r2.ordinals())
            .as("mmap-built graph must be identical to the in-RAM-built graph")
            .containsExactly(r1.ordinals());
      }

      // Ext-RaBitQ codes train + encode straight from the mmap dataset (no heap float[][]).
      VectorDataset ds = new RandomAccessVectorDataset(mbv);
      ExtendedRaBitQuantizer quantizer = ExtendedRaBitQuantizer.train(ds, 4);
      ExtendedRaBitQuantizedVectors codes = quantizer.encodeAll(ds);
      assertThat(codes.size()).isEqualTo(n);
      aMmap.enableQuantization(codes);

      int hit = 0;
      for (float[] q : queries) {
        int[] exact = brute(vecs, q, k, sim);
        hit += overlap(aMmap.search(q, k, 200, 3.0f).ordinals(), exact);
      }
      assertThat(hit / (double) (nq * k))
          .as("codes trained from the mmap dataset give good two-pass recall")
          .isGreaterThan(0.9);
    }
  }

  private static byte[] encodeVectorsBin(float[][] vectors, int dim) {
    long stride = AlignmentUtil.alignUp((long) dim * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    byte[] blob = new byte[(int) (stride * vectors.length)];
    ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < vectors.length; i++) {
      buf.position((int) (i * stride));
      for (int d = 0; d < dim; d++) {
        buf.putFloat(vectors[i][d]);
      }
    }
    return blob;
  }

  private static float[][] random(int n, int dim, long seed) {
    Random r = new Random(seed);
    float[][] v = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        v[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    return v;
  }

  private static int[] brute(float[][] d, float[] q, int k, SimilarityFunction s) {
    float[] bs = new float[k];
    int[] bi = new int[k];
    Arrays.fill(bs, Float.NEGATIVE_INFINITY);
    Arrays.fill(bi, -1);
    for (int i = 0; i < d.length; i++) {
      float sc = s.compare(q, d[i]);
      if (sc > bs[k - 1]) {
        int p = k - 1;
        while (p > 0 && bs[p - 1] < sc) {
          bs[p] = bs[p - 1];
          bi[p] = bi[p - 1];
          p--;
        }
        bs[p] = sc;
        bi[p] = i;
      }
    }
    return bi;
  }

  private static int overlap(int[] got, int[] truth) {
    Set<Integer> t = new HashSet<>();
    for (int x : truth) {
      t.add(x);
    }
    int c = 0;
    for (int x : got) {
      if (t.contains(x)) {
        c++;
      }
    }
    return c;
  }
}
