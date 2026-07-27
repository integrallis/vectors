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
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.NeighborArray;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Verifies that {@link ChunkedGraphBlob} ships a graph into many small object-storage chunks and
 * reads it back identically — the >2 GB {@code graph.bin} path — plus the legacy single-object
 * fallback.
 */
@Tag("unit")
class ChunkedGraphBlobTest {

  private static HnswGraph sampleGraph() {
    int n = 2500;
    int dim = 20;
    Random r = new Random(11L);
    float[][] vecs = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vecs[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    return HnswIndex.builder(vecs, SimilarityFunction.EUCLIDEAN)
        .maxConnections(16)
        .efConstruction(100)
        .seed(11L)
        .build()
        .graph();
  }

  @Test
  void roundTripsThroughManyChunks() throws Exception {
    HnswGraph graph = sampleGraph();
    HeapStorageBackend backend = new HeapStorageBackend();
    String prefix = "gen-0000000000000001/";

    // 777-byte chunks force the ~200 KB graph into hundreds of objects.
    long total = ChunkedGraphBlob.writeGraph(backend, prefix, graph, 777);

    long expected = HnswGraphCodec.encode(graph).length;
    assertThat(total).as("reported total bytes").isEqualTo(expected);

    long chunkObjects =
        backend.list(prefix).stream().filter(k -> k.contains(FileFormat.GRAPH_FILE + ".")).count();
    assertThat(chunkObjects).as("multiple chunk objects written").isGreaterThan(1);

    HnswGraph decoded = ChunkedGraphBlob.openGraph(backend, prefix);
    assertGraphsEqual(graph, decoded);
  }

  @Test
  void readsLegacySingleObjectGraphBin() throws Exception {
    HnswGraph graph = sampleGraph();
    HeapStorageBackend backend = new HeapStorageBackend();
    String prefix = "gen-legacy/";
    // Write the old way: one graph.bin object, no chunk suffix.
    backend.put(prefix + FileFormat.GRAPH_FILE, HnswGraphCodec.encode(graph));

    HnswGraph decoded = ChunkedGraphBlob.openGraph(backend, prefix);
    assertGraphsEqual(graph, decoded);
  }

  @Test
  void returnsNullWhenAbsent() throws Exception {
    assertThat(ChunkedGraphBlob.openGraph(new HeapStorageBackend(), "gen-empty/")).isNull();
  }

  private static void assertGraphsEqual(HnswGraph a, HnswGraph b) {
    assertThat(b.size()).isEqualTo(a.size());
    assertThat(b.maxLevel()).isEqualTo(a.maxLevel());
    assertThat(b.entryNode()).isEqualTo(a.entryNode());
    for (int i = 0; i < a.size(); i++) {
      assertThat(b.nodeLevel(i)).isEqualTo(a.nodeLevel(i));
      for (int layer = 0; layer <= a.nodeLevel(i); layer++) {
        NeighborArray na = a.getNeighbors(i, layer);
        NeighborArray nb = b.getNeighbors(i, layer);
        assertThat(nb.size()).isEqualTo(na.size());
        for (int k = 0; k < na.size(); k++) {
          assertThat(nb.node(k)).isEqualTo(na.node(k));
        }
      }
    }
  }
}
