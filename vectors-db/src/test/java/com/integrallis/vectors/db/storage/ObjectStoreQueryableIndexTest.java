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
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizer;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ObjectStoreQueryableIndexTest {

  @Test
  void queriesShippedGenerationDirectlyFromObjectStorage() throws Exception {
    int n = 2000;
    int dim = 64;
    int k = 10;
    int nq = 30;
    float over = 3.0f;
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

    float[][] vecs = random(n, dim, 1L);
    float[][] queries = random(nq, dim, 2L);

    HnswGraph graph =
        HnswIndex.builder(vecs, sim).maxConnections(32).efConstruction(200).build().graph();
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    ExtendedRaBitQuantizer quantizer = ExtendedRaBitQuantizer.train(ds, 4);
    ExtendedRaBitQuantizedVectors codes = quantizer.encodeAll(ds);

    // "Ship" a generation to object storage in the GenerationShippingSubscriber layout.
    HeapStorageBackend backend = new HeapStorageBackend();
    long gen = 7L;
    String gp = FileFormat.generationDirName(gen) + "/";
    backend.put(gp + FileFormat.GRAPH_FILE, HnswGraphCodec.encode(graph));
    backend.put(
        gp + FileFormat.QUANTIZED_FILE,
        QuantizedVectorsCodec.encode(codes, quantizer, QuantizerKind.EXTENDED_RABITQ));
    backend.put(gp + FileFormat.VECTORS_FILE, encodeVectorsBin(vecs, dim));
    backend.put(
        FileFormat.CURRENT_FILE,
        ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(gen).array());

    // Baseline: same graph + codes, full vectors in RAM.
    MappedHnswIndexAdapter local =
        new MappedHnswIndexAdapter(graph, new InMemoryVectors(vecs), sim);
    local.enableQuantization(codes);

    try (ObjectStoreQueryableIndex idx = ObjectStoreQueryableIndex.openCurrent(backend, "", sim)) {
      assertThat(idx.size()).isEqualTo(n);
      assertThat(idx.dimension()).isEqualTo(dim);

      int match = 0;
      int hit = 0;
      for (float[] q : queries) {
        int[] exact = brute(vecs, q, k, sim);
        SearchOutcome lo = local.search(q, k, 100, over);
        SearchOutcome ro = idx.search(q, k, 100, over);
        if (Arrays.equals(lo.ordinals(), ro.ordinals())) {
          match++;
        }
        hit += overlap(ro.ordinals(), exact);
      }
      assertThat(match)
          .as("index opened straight from object storage must match the in-RAM path")
          .isEqualTo(nq);
      assertThat(hit / (double) (nq * k)).as("recall@10 over object storage").isGreaterThan(0.95);
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
