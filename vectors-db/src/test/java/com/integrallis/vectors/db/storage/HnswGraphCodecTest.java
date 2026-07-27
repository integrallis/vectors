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
import static org.assertj.core.api.Assertions.assertThatIOException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.NeighborArray;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class HnswGraphCodecTest {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /**
   * Builds a small, deterministic graph for round-trip tests. Every node lives at layer 0, a few
   * nodes also at layer 1 and layer 2. Neighbors are assigned by position (cyclic) with descending
   * synthetic scores so the resulting NeighborArrays iterate in a known order.
   */
  private static HnswGraph buildSmallGraph(int numNodes, int m, long seed) {
    HnswGraph graph = new HnswGraph(numNodes, m);
    Random rng = new Random(seed);
    int[] levels = new int[numNodes];
    int maxLevel = 0;
    for (int i = 0; i < numNodes; i++) {
      // 10% at layer 2, 25% at layer 1, the rest at layer 0.
      double r = rng.nextDouble();
      int level = r < 0.10 ? 2 : (r < 0.35 ? 1 : 0);
      levels[i] = level;
      if (level > maxLevel) {
        maxLevel = level;
      }
      graph.initNode(i, level);
    }
    // Guarantee at least one layer-0 edge between every pair of neighbors so every node has a
    // non-empty adjacency list. We connect each node i to (i+1) % n and (i+2) % n (cycle of
    // length 3 at least for n>=3).
    for (int i = 0; i < numNodes; i++) {
      NeighborArray arr = graph.getNeighbors(i, 0);
      int limit = Math.min(2 * m, Math.max(0, numNodes - 1));
      for (int step = 1; step <= limit; step++) {
        int nbr = (i + step) % numNodes;
        if (nbr == i) {
          continue;
        }
        // Deterministic descending scores so the insertion is O(1) tail-append.
        arr.insert(nbr, (float) (limit - step + 1));
      }
    }
    // Add upper layer edges for nodes at level >= 1
    for (int layer = 1; layer <= maxLevel; layer++) {
      for (int i = 0; i < numNodes; i++) {
        if (levels[i] < layer) {
          continue;
        }
        NeighborArray arr = graph.getNeighbors(i, layer);
        int edges = 0;
        // Scan in order for other nodes at this layer.
        for (int j = 0; j < numNodes && edges < m; j++) {
          if (j == i || levels[j] < layer) {
            continue;
          }
          arr.insert(j, (float) (m - edges));
          edges++;
        }
      }
    }
    // Set entry to the highest-level node we saw first.
    int entry = -1;
    for (int i = 0; i < numNodes; i++) {
      if (levels[i] == maxLevel) {
        entry = i;
        break;
      }
    }
    if (entry >= 0) {
      graph.setEntryNode(entry, maxLevel);
    }
    return graph;
  }

  private static void assertGraphsEqualStructurally(HnswGraph expected, HnswGraph actual) {
    assertThat(actual.size()).as("size").isEqualTo(expected.size());
    assertThat(actual.maxConnections()).as("maxConnections").isEqualTo(expected.maxConnections());
    assertThat(actual.entryNode()).as("entryNode").isEqualTo(expected.entryNode());
    assertThat(actual.maxLevel()).as("maxLevel").isEqualTo(expected.maxLevel());

    int n = expected.size();
    for (int i = 0; i < n; i++) {
      assertThat(actual.nodeLevel(i)).as("nodeLevel[" + i + "]").isEqualTo(expected.nodeLevel(i));
    }
    for (int i = 0; i < n; i++) {
      assertLayerEqual(expected, actual, i, 0);
      for (int layer = 1; layer <= expected.nodeLevel(i); layer++) {
        assertLayerEqual(expected, actual, i, layer);
      }
    }
  }

  private static void assertLayerEqual(HnswGraph expected, HnswGraph actual, int node, int layer) {
    NeighborArray e = expected.getNeighbors(node, layer);
    NeighborArray a = actual.getNeighbors(node, layer);
    assertThat(a).as("layer " + layer + " node " + node + " neighbors present").isNotNull();
    assertThat(a.size()).as("layer " + layer + " node " + node + " size").isEqualTo(e.size());
    for (int k = 0; k < e.size(); k++) {
      assertThat(a.node(k))
          .as("layer " + layer + " node " + node + " neighbor[" + k + "]")
          .isEqualTo(e.node(k));
    }
  }

  // ---------------------------------------------------------------------------
  // Round-trip tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class RoundTrip {

    @Test
    void emptyGraphRoundTrips() throws IOException {
      HnswGraph empty = new HnswGraph(1, 16);
      byte[] bytes = HnswGraphCodec.encode(empty);
      assertThat(bytes).hasSize(HnswGraphCodec.HEADER_SIZE);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      assertThat(decoded.size()).isZero();
      assertThat(decoded.maxConnections()).isEqualTo(16);
      assertThat(decoded.entryNode()).isEqualTo(-1);
      assertThat(decoded.maxLevel()).isEqualTo(-1);
    }

    @Test
    void singleNodeGraphRoundTrips() throws IOException {
      HnswGraph single = new HnswGraph(1, 4);
      single.initNode(0, 0);
      single.setEntryNode(0, 0);

      byte[] bytes = HnswGraphCodec.encode(single);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      assertThat(decoded.size()).isEqualTo(1);
      assertThat(decoded.entryNode()).isZero();
      assertThat(decoded.maxLevel()).isZero();
      assertThat(decoded.nodeLevel(0)).isZero();
      assertThat(decoded.getNeighbors(0, 0)).isNotNull();
      assertThat(decoded.getNeighbors(0, 0).size()).isZero();
    }

    @Test
    void smallGraphRoundTripsExact() throws IOException {
      HnswGraph original = buildSmallGraph(100, 8, 42L);
      byte[] bytes = HnswGraphCodec.encode(original);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void mediumGraphWithM16RoundTrips() throws IOException {
      HnswGraph original = buildSmallGraph(500, 16, 7L);
      byte[] bytes = HnswGraphCodec.encode(original);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void mediumGraphWithM32RoundTrips() throws IOException {
      HnswGraph original = buildSmallGraph(200, 32, 13L);
      byte[] bytes = HnswGraphCodec.encode(original);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void reusedEncodeCallsProduceIdenticalBytes() {
      // Quality-checklist "reused buffer" test: encoding the same graph twice must produce the
      // same byte array. Guards against accumulator reuse or state leakage in encode().
      HnswGraph graph = buildSmallGraph(50, 8, 3L);
      byte[] first = HnswGraphCodec.encode(graph);
      byte[] second = HnswGraphCodec.encode(graph);
      assertThat(second).isEqualTo(first);
    }

    @Test
    void decodedGraphPreservesLayer0NeighborOrder() throws IOException {
      // Build a known graph where we control insertion order at layer 0.
      HnswGraph graph = new HnswGraph(5, 4);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.initNode(3, 0);
      graph.initNode(4, 0);
      graph.setEntryNode(0, 0);

      // Insert with strictly decreasing scores so positions are deterministic.
      NeighborArray n0 = graph.getNeighbors(0, 0);
      n0.insert(3, 100f);
      n0.insert(4, 50f);
      n0.insert(1, 10f);

      byte[] bytes = HnswGraphCodec.encode(graph);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);

      NeighborArray d0 = decoded.getNeighbors(0, 0);
      assertThat(d0.size()).isEqualTo(3);
      assertThat(d0.node(0)).isEqualTo(3);
      assertThat(d0.node(1)).isEqualTo(4);
      assertThat(d0.node(2)).isEqualTo(1);
    }

    @Test
    void decodedGraphRejectsMisalignedLayerNodes() throws IOException {
      // This is the inverse check — a correct round-trip should never violate the "nodeId appears
      // only at layers <= nodeLevels[nodeId]" invariant, even for graphs where upper-layer nodes
      // are sparse.
      HnswGraph graph = buildSmallGraph(30, 4, 99L);
      byte[] bytes = HnswGraphCodec.encode(graph);
      HnswGraph decoded = HnswGraphCodec.decode(bytes);
      // If any upper-layer node appears at a layer > its nodeLevels, decode() would have thrown.
      // Instead we simply assert the graph round-trips cleanly.
      assertGraphsEqualStructurally(graph, decoded);
    }
  }

  // ---------------------------------------------------------------------------
  // Header-validation tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class HeaderValidation {

    private byte[] encodeSample() {
      return HnswGraphCodec.encode(buildSmallGraph(20, 4, 1L));
    }

    @Test
    void rejectsTruncatedHeader() {
      byte[] tooShort = new byte[HnswGraphCodec.HEADER_SIZE - 1];
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(tooShort))
          .withMessageContaining("truncated");
    }

    @Test
    void rejectsWrongMagic() {
      byte[] encoded = encodeSample();
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void rejectsWrongVersion() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("version");
    }

    @Test
    void rejectsWrongHeaderLength() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(8, 999);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("header length");
    }

    @Test
    void rejectsNonZeroFlags() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(12, 1);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("flags");
    }

    @Test
    void rejectsNegativeNumNodes() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(16, -1);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("numNodes");
    }

    @Test
    void rejectsZeroMaxConnections() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(20, 0);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("maxConnections");
    }

    @Test
    void rejectsEntryNodeOutOfRange() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      int numNodes = buf.getInt(16);
      buf.putInt(24, numNodes); // out of range
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("entryNode");
    }

    @Test
    void rejectsNonEmptyGraphWithMaxLevelMinusOne() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(28, -1);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("maxLevel");
    }

    @Test
    void rejectsEmptyGraphWithEntryNodeNotMinusOne() {
      // Synthesize an empty graph header with bogus entryNode.
      HnswGraph empty = new HnswGraph(1, 16);
      byte[] encoded = HnswGraphCodec.encode(empty);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(24, 0); // entryNode = 0 but numNodes = 0
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("empty graph");
    }
  }

  // ---------------------------------------------------------------------------
  // Body-validation tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class BodyValidation {

    @Test
    void rejectsTruncatedNodeLevels() {
      byte[] encoded = HnswGraphCodec.encode(buildSmallGraph(10, 4, 2L));
      // Shrink the buffer to HEADER + 2 bytes of nodeLevels (far too short).
      byte[] truncated = new byte[HnswGraphCodec.HEADER_SIZE + 2];
      System.arraycopy(encoded, 0, truncated, 0, truncated.length);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(truncated))
          .withMessageContaining("nodeLevels");
    }

    @Test
    void rejectsNodeLevelExceedingMaxLevel() {
      // Build a graph with maxLevel = 0 (everyone at layer 0), then poison one nodeLevels entry.
      HnswGraph flat = new HnswGraph(5, 4);
      for (int i = 0; i < 5; i++) {
        flat.initNode(i, 0);
      }
      flat.setEntryNode(0, 0);
      byte[] encoded = HnswGraphCodec.encode(flat);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // nodeLevels starts at offset 32. Overwrite entry 0 with level 99.
      buf.putInt(32, 99);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("nodeLevels");
    }

    @Test
    void rejectsLayer0DegreeExceedingCapacity() throws IOException {
      // Construct a tiny valid graph, then rewrite the first layer-0 degree word to 2*M + 5.
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 0);
      graph.getNeighbors(0, 0).insert(1, 10f);
      graph.getNeighbors(0, 0).insert(2, 5f);

      byte[] encoded = HnswGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // nodeLevels is 3 * 4 = 12 bytes. Layer 0 adjacency starts at offset 32 + 12 = 44.
      buf.putInt(44, 2 * 2 + 5); // 2*M=4, so any value > 4 is a violation.
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("degree");
    }

    @Test
    void rejectsLayer0NeighborOutOfRange() throws IOException {
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 0);
      graph.getNeighbors(0, 0).insert(1, 10f);
      byte[] encoded = HnswGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // Layer 0 starts at offset 44. First entry: degree=1 (4 bytes) then neighbor id (4 bytes).
      // Overwrite the neighbor id with something out of range.
      buf.putInt(44 + 4, 99);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("out of range");
    }

    @Test
    void rejectsSelfLoopNeighbor() throws IOException {
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 0);
      graph.getNeighbors(0, 0).insert(1, 10f);
      byte[] encoded = HnswGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // Layer 0 first neighbor at offset 44+4 = 48. Rewrite to 0 (self-loop).
      buf.putInt(48, 0);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("self-loop");
    }

    @Test
    void rejectsTrailingBytes() {
      byte[] encoded = HnswGraphCodec.encode(buildSmallGraph(5, 2, 3L));
      byte[] padded = new byte[encoded.length + 7];
      System.arraycopy(encoded, 0, padded, 0, encoded.length);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(padded))
          .withMessageContaining("trailing");
    }

    @Test
    void rejectsEntryNodeBelowMaxLevel() {
      // HNSW invariant: the entry node must live at the topmost layer (maxLevel). A graph where
      // nodeLevels[entryNode] < maxLevel is silently wrong at search time (greedy descent would
      // start below the top and miss the seed connections), so decode() must refuse it.
      //
      // Build a small graph with entryNode=0 at level 2, then poison nodeLevels[0] to 1.
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 2);
      graph.initNode(1, 2);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 2);
      graph.getNeighbors(0, 0).insert(1, 3f);
      graph.getNeighbors(0, 0).insert(2, 2f);
      graph.getNeighbors(1, 0).insert(0, 3f);
      graph.getNeighbors(2, 0).insert(0, 2f);
      graph.getNeighbors(0, 1).insert(1, 2f);
      graph.getNeighbors(1, 1).insert(0, 2f);
      graph.getNeighbors(0, 2).insert(1, 1f);
      graph.getNeighbors(1, 2).insert(0, 1f);

      byte[] encoded = HnswGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // nodeLevels[0] lives at offset HEADER_SIZE = 32. Overwrite with level 1 (< maxLevel=2).
      buf.putInt(32, 1);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(encoded))
          .withMessageContaining("entryNode");
    }
  }

  // ---------------------------------------------------------------------------
  // Encode-time validation tests (defense-in-depth against corrupt graph state)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class EncodeValidation {

    @Test
    void encodeRejectsSelfLoopAtLayerZero() {
      // NeighborArray.insert() does not block inserting node i as its own neighbor, so a buggy
      // or malicious builder could produce a self-loop. encode() must refuse it on the way out
      // so the corruption never reaches graph.bin.
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 0);
      graph.getNeighbors(0, 0).insert(0, 10f); // self-loop
      assertThatIllegalStateException()
          .isThrownBy(() -> HnswGraphCodec.encode(graph))
          .withMessageContaining("self-loop");
    }

    @Test
    void encodeRejectsOutOfRangeNeighborAtLayerZero() {
      // NeighborArray.insert() does not bounds-check the nodeId against numNodes, so an off-by-one
      // builder bug could persist a stale ordinal. encode() must catch it before it reaches disk.
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 0);
      graph.getNeighbors(0, 0).insert(99, 10f); // out of range
      assertThatIllegalStateException()
          .isThrownBy(() -> HnswGraphCodec.encode(graph))
          .withMessageContaining("out of range");
    }

    @Test
    void encodeRejectsSelfLoopAtUpperLayer() {
      // The upper-layer path has its own neighbor-validation loop; verify the self-loop guard
      // fires there too (not just at layer 0).
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 1);
      graph.initNode(1, 1);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 1);
      graph.getNeighbors(0, 0).insert(1, 3f);
      graph.getNeighbors(0, 0).insert(2, 2f);
      graph.getNeighbors(1, 0).insert(0, 3f);
      graph.getNeighbors(2, 0).insert(0, 2f);
      graph.getNeighbors(0, 1).insert(0, 10f); // self-loop at layer 1
      assertThatIllegalStateException()
          .isThrownBy(() -> HnswGraphCodec.encode(graph))
          .withMessageContaining("self-loop");
    }

    @Test
    void encodeRejectsOutOfRangeNeighborAtUpperLayer() {
      HnswGraph graph = new HnswGraph(3, 2);
      graph.initNode(0, 1);
      graph.initNode(1, 1);
      graph.initNode(2, 0);
      graph.setEntryNode(0, 1);
      graph.getNeighbors(0, 0).insert(1, 3f);
      graph.getNeighbors(0, 0).insert(2, 2f);
      graph.getNeighbors(1, 0).insert(0, 3f);
      graph.getNeighbors(2, 0).insert(0, 2f);
      graph.getNeighbors(0, 1).insert(99, 10f); // out of range at layer 1
      assertThatIllegalStateException()
          .isThrownBy(() -> HnswGraphCodec.encode(graph))
          .withMessageContaining("out of range");
    }
  }

  // ---------------------------------------------------------------------------
  // Null / API-contract tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class NullGuard {

    @Test
    void encodeRejectsNullGraph() {
      assertThatNullPointerException().isThrownBy(() -> HnswGraphCodec.encode(null));
    }

    @Test
    void decodeRejectsNullBytes() {
      assertThatNullPointerException().isThrownBy(() -> HnswGraphCodec.decode((byte[]) null));
    }

    @Test
    void emptyGraphWithSetEntryNodeEncodedThenDecodedRejectedAsInconsistent() {
      // setEntryNode can be called on a graph with no initialized nodes, yielding an inconsistent
      // degenerate state that encode() happily writes but decode() must refuse. This is the
      // closest analogue to the "encode of an inconsistent graph" case accessible from public
      // API.
      HnswGraph broken = new HnswGraph(1, 4);
      broken.setEntryNode(0, 0);
      byte[] bytes = HnswGraphCodec.encode(broken);
      assertThatIOException()
          .isThrownBy(() -> HnswGraphCodec.decode(bytes))
          .withMessageContaining("empty graph");
    }
  }
}
