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
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Proves the streaming {@code graph.bin} codec ({@link HnswGraphCodec#encode(HnswGraph,
 * java.io.OutputStream)} / {@link HnswGraphCodec#decode(InputStream)}) produces byte-identical
 * output to the {@code byte[]} codec and round-trips a graph without ever materializing the whole
 * image as one array — the property that lets a >2 GB graph serialize.
 */
@Tag("unit")
class HnswGraphCodecStreamingTest {

  private static HnswGraph sampleGraph() {
    int n = 3000;
    int dim = 24;
    Random r = new Random(7L);
    float[][] vecs = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vecs[i][d] = r.nextFloat() * 2f - 1f;
      }
    }
    return HnswIndex.builder(vecs, SimilarityFunction.EUCLIDEAN)
        .maxConnections(16)
        .efConstruction(100)
        .seed(7L)
        .build()
        .graph();
  }

  @Test
  void streamingEncodeIsByteIdenticalToArrayEncode() throws Exception {
    HnswGraph graph = sampleGraph();
    byte[] arrayForm = HnswGraphCodec.encode(graph);

    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    long written = HnswGraphCodec.encode(graph, bos);

    assertThat(written).as("reported length").isEqualTo(arrayForm.length);
    assertThat(bos.toByteArray()).as("streaming bytes == array bytes").isEqualTo(arrayForm);
  }

  @Test
  void streamingDecodeMatchesArrayDecode() throws Exception {
    HnswGraph graph = sampleGraph();
    byte[] form = HnswGraphCodec.encode(graph);

    HnswGraph viaArray = HnswGraphCodec.decode(form);
    HnswGraph viaStream = HnswGraphCodec.decode(new ByteArrayInputStream(form));
    assertGraphsEqual(viaArray, viaStream);
    assertGraphsEqual(graph, viaStream);
  }

  @Test
  void decodesFromAChunkedStream() throws Exception {
    // Split the encoded image into many tiny chunks and feed them as a concatenated stream. This
    // is the object-storage shape (graph.bin.000, .001, ...) and proves decode never depends on a
    // single contiguous buffer — the core property for a >2 GB graph.
    HnswGraph graph = sampleGraph();
    byte[] form = HnswGraphCodec.encode(graph);

    int chunk = 997; // deliberately not a multiple of 4, so int reads straddle chunk boundaries
    List<InputStream> parts = new ArrayList<>();
    for (int off = 0; off < form.length; off += chunk) {
      int len = Math.min(chunk, form.length - off);
      byte[] slice = new byte[len];
      System.arraycopy(form, off, slice, 0, len);
      parts.add(new ByteArrayInputStream(slice));
    }
    try (InputStream seq = new SequenceInputStream(Collections.enumeration(parts))) {
      HnswGraph decoded = HnswGraphCodec.decode(seq);
      assertGraphsEqual(graph, decoded);
    }
  }

  private static void assertGraphsEqual(HnswGraph a, HnswGraph b) {
    assertThat(b.size()).isEqualTo(a.size());
    assertThat(b.maxLevel()).isEqualTo(a.maxLevel());
    assertThat(b.entryNode()).isEqualTo(a.entryNode());
    assertThat(b.maxConnections()).isEqualTo(a.maxConnections());
    for (int i = 0; i < a.size(); i++) {
      assertThat(b.nodeLevel(i)).as("level of node " + i).isEqualTo(a.nodeLevel(i));
      for (int layer = 0; layer <= a.nodeLevel(i); layer++) {
        NeighborArray na = a.getNeighbors(i, layer);
        NeighborArray nb = b.getNeighbors(i, layer);
        assertThat(nb.size()).as("degree node " + i + " layer " + layer).isEqualTo(na.size());
        for (int k = 0; k < na.size(); k++) {
          assertThat(nb.node(k))
              .as("neighbor " + k + " of node " + i + " layer " + layer)
              .isEqualTo(na.node(k));
        }
      }
    }
  }
}
