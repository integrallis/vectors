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
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizedVectors;
import com.integrallis.vectors.quantization.ExtendedRaBitQuantizer;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end proof of the object-storage query path: the SAME decoded {@link HnswGraph} +
 * Ext-RaBitQ codes are searched two ways — full vectors in RAM ({@link InMemoryVectors}) vs served
 * over a {@link StorageBackend} via ranged GET ({@link ObjectStoreRandomAccessVectors}). Because
 * navigation runs entirely on the RAM-resident codes and the only object-storage reads are the
 * rerank fetches, the two paths must return identical top-k, and the object-storage path must issue
 * only a small, bounded number of GETs per query (≈ the over-query rerank set), never a scan of all
 * N.
 */
@Tag("unit")
class ObjectStoreQueryPathTest {

  /** StorageBackend decorator that counts ranged GETs (the per-query object-storage read cost). */
  private static final class CountingBackend implements StorageBackend {
    private final StorageBackend inner;
    final AtomicLong getRangeCalls = new AtomicLong();

    CountingBackend(StorageBackend inner) {
      this.inner = inner;
    }

    @Override
    public void put(String key, byte[] value) throws IOException {
      inner.put(key, value);
    }

    @Override
    public byte[] get(String key) throws IOException {
      return inner.get(key);
    }

    @Override
    public StoredValue getWithEtag(String key) throws IOException {
      return inner.getWithEtag(key);
    }

    @Override
    public byte[] getRange(String key, long offset, int length) throws IOException {
      getRangeCalls.incrementAndGet();
      return inner.getRange(key, offset, length);
    }

    @Override
    public List<String> list(String prefix) throws IOException {
      return inner.list(prefix);
    }

    @Override
    public void delete(String key) throws IOException {
      inner.delete(key);
    }

    @Override
    public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
        throws IOException {
      return inner.conditionalPut(key, value, expectedEtag);
    }
  }

  private static byte[] encodeVectorsBin(float[][] vectors, int dim) {
    long stride =
        com.integrallis.vectors.storage.memory.AlignmentUtil.alignUp(
            (long) dim * Float.BYTES,
            com.integrallis.vectors.storage.memory.AlignmentUtil.VECTOR_ALIGNMENT);
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

  @Test
  void objectStorePathMatchesInMemoryAndBoundsGetsPerQuery() throws Exception {
    int n = 2000;
    int dim = 64;
    int k = 10;
    int nq = 50;
    float overQuery = 3.0f;
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;

    float[][] vecs = random(n, dim, 1L);
    float[][] queries = random(nq, dim, 2L);

    // Build the graph once; both adapters share this exact topology.
    HnswGraph graph =
        HnswIndex.builder(vecs, sim).maxConnections(16).efConstruction(200).build().graph();

    // Shared RAM-resident Ext-RaBitQ codes (navigation tier).
    ArrayVectorDataset ds = new ArrayVectorDataset(vecs);
    ExtendedRaBitQuantizedVectors codes = ExtendedRaBitQuantizer.train(ds, 4).encodeAll(ds);

    // Full vectors served over object storage (ranged GET), with a GET counter.
    CountingBackend backend = new CountingBackend(new HeapStorageBackend());
    backend.put("vectors.bin", encodeVectorsBin(vecs, dim));

    MappedHnswIndexAdapter local =
        new MappedHnswIndexAdapter(graph, new InMemoryVectors(vecs), sim);
    local.enableQuantization(codes);

    MappedHnswIndexAdapter remote =
        new MappedHnswIndexAdapter(
            graph, new ObjectStoreRandomAccessVectors(backend, "vectors.bin", n, dim), sim);
    remote.enableQuantization(codes);

    long totalGets = 0;
    for (float[] query : queries) {
      SearchOutcome expected = local.search(query, k, 100, overQuery);
      long before = backend.getRangeCalls.get();
      SearchOutcome actual = remote.search(query, k, 100, overQuery);
      long gets = backend.getRangeCalls.get() - before;
      totalGets += gets;

      assertThat(actual.ordinals())
          .as("object-storage top-k must equal in-memory top-k")
          .containsExactly(expected.ordinals());
      assertThat(gets).as("some rerank vectors must be fetched").isPositive();
      assertThat(gets).as("GETs/query must be << N (no full scan)").isLessThan(n / 4);
    }

    double avgGets = totalGets / (double) nq;
    System.out.printf(
        "object-storage query path: avg %.1f GETs/query (n=%d, ef=100, rerank=%.0fx)%n",
        avgGets, n, overQuery);
    // Rerank set ≈ overQueryFactor * k ≈ 30; navigation adds no GETs (runs on RAM codes).
    assertThat(avgGets).isLessThan(200.0);
  }
}
