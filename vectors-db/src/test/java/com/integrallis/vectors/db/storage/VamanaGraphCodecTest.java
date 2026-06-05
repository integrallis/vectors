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

import com.integrallis.vectors.vamana.NeighborArray;
import com.integrallis.vectors.vamana.VamanaGraph;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class VamanaGraphCodecTest {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /**
   * Builds a small, deterministic Vamana graph for round-trip tests. Every node gets up to {@code
   * degreeLimit} neighbors, assigned by position (cyclic) with descending synthetic scores so the
   * resulting NeighborArrays iterate in a known order. The medoid is always node 0.
   */
  private static VamanaGraph buildSmallGraph(int numNodes, int maxDegree, long seed) {
    VamanaGraph graph = new VamanaGraph(numNodes, maxDegree);
    Random rng = new Random(seed);
    for (int i = 0; i < numNodes; i++) {
      graph.initNode(i);
    }
    // Give every node some neighbors but stay strictly below maxDegree so encode() does not
    // reject the graph. We pick deterministic neighbors starting from (i+1) with a random
    // jitter offset so the per-node neighbor sets are not all identical.
    int target = Math.min(maxDegree, Math.max(1, numNodes - 1));
    for (int i = 0; i < numNodes; i++) {
      NeighborArray arr = graph.getNeighbors(i);
      int jitter = rng.nextInt(Math.max(1, numNodes));
      for (int step = 1; step <= target; step++) {
        int nbr = ((i + step + jitter) % numNodes);
        if (nbr == i) {
          // Shift by one more to avoid self-loop. If the shifted value still collides with i
          // (only possible when numNodes == 1 which we excluded above) skip this slot.
          nbr = (nbr + 1) % numNodes;
          if (nbr == i) {
            continue;
          }
        }
        // Deterministic descending scores so insertion is O(1) tail-append.
        arr.insert(nbr, (float) (target - step + 1));
      }
    }
    graph.setMedoid(0);
    return graph;
  }

  private static void assertGraphsEqualStructurally(VamanaGraph expected, VamanaGraph actual) {
    assertThat(actual.size()).as("size").isEqualTo(expected.size());
    assertThat(actual.maxDegree()).as("maxDegree").isEqualTo(expected.maxDegree());
    // medoid is only meaningful for non-empty graphs; for empty graphs we synthesize -1 on
    // encode and the decoded graph's default medoid is 0, so we skip the check when empty.
    if (expected.size() > 0) {
      assertThat(actual.medoid()).as("medoid").isEqualTo(expected.medoid());
    }

    int n = expected.size();
    for (int i = 0; i < n; i++) {
      NeighborArray e = expected.getNeighbors(i);
      NeighborArray a = actual.getNeighbors(i);
      assertThat(a).as("node " + i + " neighbors present").isNotNull();
      assertThat(a.size()).as("node " + i + " size").isEqualTo(e.size());
      for (int k = 0; k < e.size(); k++) {
        assertThat(a.node(k)).as("node " + i + " neighbor[" + k + "]").isEqualTo(e.node(k));
      }
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
      VamanaGraph empty = new VamanaGraph(1, 16);
      byte[] bytes = VamanaGraphCodec.encode(empty);
      assertThat(bytes).hasSize(VamanaGraphCodec.HEADER_SIZE);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertThat(decoded.size()).isZero();
      assertThat(decoded.maxDegree()).isEqualTo(16);
    }

    @Test
    void singleNodeGraphRoundTrips() throws IOException {
      VamanaGraph single = new VamanaGraph(1, 4);
      single.initNode(0);
      single.setMedoid(0);

      byte[] bytes = VamanaGraphCodec.encode(single);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertThat(decoded.size()).isEqualTo(1);
      assertThat(decoded.medoid()).isZero();
      assertThat(decoded.getNeighbors(0)).isNotNull();
      assertThat(decoded.getNeighbors(0).size()).isZero();
    }

    @Test
    void smallGraphRoundTripsExact() throws IOException {
      VamanaGraph original = buildSmallGraph(100, 8, 42L);
      byte[] bytes = VamanaGraphCodec.encode(original);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void mediumGraphWithR16RoundTrips() throws IOException {
      VamanaGraph original = buildSmallGraph(500, 16, 7L);
      byte[] bytes = VamanaGraphCodec.encode(original);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void mediumGraphWithR32RoundTrips() throws IOException {
      VamanaGraph original = buildSmallGraph(200, 32, 13L);
      byte[] bytes = VamanaGraphCodec.encode(original);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertGraphsEqualStructurally(original, decoded);
    }

    @Test
    void reusedEncodeCallsProduceIdenticalBytes() {
      // Quality-checklist "reused buffer" test: encoding the same graph twice must produce the
      // same byte array. Guards against accumulator reuse or state leakage in encode().
      VamanaGraph graph = buildSmallGraph(50, 8, 3L);
      byte[] first = VamanaGraphCodec.encode(graph);
      byte[] second = VamanaGraphCodec.encode(graph);
      assertThat(second).isEqualTo(first);
    }

    @Test
    void decodedGraphPreservesNeighborOrder() throws IOException {
      // Build a known graph where we control insertion order; decode must preserve the exact
      // descending-score iteration order on the other side.
      VamanaGraph graph = new VamanaGraph(5, 4);
      for (int i = 0; i < 5; i++) {
        graph.initNode(i);
      }
      graph.setMedoid(0);

      // Insert with strictly decreasing scores so positions are deterministic.
      NeighborArray n0 = graph.getNeighbors(0);
      n0.insert(3, 100f);
      n0.insert(4, 50f);
      n0.insert(1, 10f);

      byte[] bytes = VamanaGraphCodec.encode(graph);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);

      NeighborArray d0 = decoded.getNeighbors(0);
      assertThat(d0.size()).isEqualTo(3);
      assertThat(d0.node(0)).isEqualTo(3);
      assertThat(d0.node(1)).isEqualTo(4);
      assertThat(d0.node(2)).isEqualTo(1);
    }

    @Test
    void nonZeroMedoidRoundTrips() throws IOException {
      VamanaGraph graph = buildSmallGraph(20, 4, 99L);
      graph.setMedoid(7);
      byte[] bytes = VamanaGraphCodec.encode(graph);
      VamanaGraph decoded = VamanaGraphCodec.decode(bytes);
      assertThat(decoded.medoid()).isEqualTo(7);
    }
  }

  // ---------------------------------------------------------------------------
  // Header-validation tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class HeaderValidation {

    private byte[] encodeSample() {
      return VamanaGraphCodec.encode(buildSmallGraph(20, 4, 1L));
    }

    @Test
    void rejectsTruncatedHeader() {
      byte[] tooShort = new byte[VamanaGraphCodec.HEADER_SIZE - 1];
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(tooShort))
          .withMessageContaining("truncated");
    }

    @Test
    void rejectsWrongMagic() {
      byte[] encoded = encodeSample();
      encoded[0] ^= (byte) 0xFF;
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void rejectsHnswMagicMistakenlyFedToVamanaDecoder() {
      // A file that starts with the HNSW magic "VGPH" must be rejected at the very first check
      // when the caller mistakenly routes it to VamanaGraphCodec.decode. This is the key safety
      // net that distinct magics give us — the alternative would be a silent wrong-shaped decode.
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(0, FileFormat.MAGIC_GRAPH); // HNSW magic
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("magic");
    }

    @Test
    void rejectsWrongVersion() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(4, 999);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("version");
    }

    @Test
    void rejectsWrongHeaderLength() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(8, 999);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("header length");
    }

    @Test
    void rejectsNonZeroFlags() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(12, 1);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("flags");
    }

    @Test
    void rejectsNegativeNumNodes() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(16, -1);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("numNodes");
    }

    @Test
    void rejectsZeroMaxDegree() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(20, 0);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("maxDegree");
    }

    @Test
    void rejectsMedoidOutOfRange() {
      byte[] encoded = encodeSample();
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      int numNodes = buf.getInt(16);
      buf.putInt(24, numNodes); // medoid == numNodes, out of range
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("medoid");
    }

    @Test
    void rejectsEmptyGraphWithMedoidNotMinusOne() {
      // Synthesize an empty-graph header with a bogus (non-sentinel) medoid.
      VamanaGraph empty = new VamanaGraph(1, 16);
      byte[] encoded = VamanaGraphCodec.encode(empty);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(24, 0); // medoid = 0 but numNodes = 0
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("empty graph");
    }
  }

  // ---------------------------------------------------------------------------
  // Offset-index (v2) tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class OffsetIndex {

    @Test
    void encodedFileCarriesValidOffsetTrailer() throws IOException {
      VamanaGraph graph = buildSmallGraph(64, 8, 5L);
      byte[] bytes = VamanaGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

      int numNodes = buf.getInt(16);
      assertThat(numNodes).isEqualTo(64);

      // File = header + body + 8*N trailer; the trailer starts at fileLen - 8N.
      long trailerStart = bytes.length - 8L * numNodes;
      assertThat(trailerStart).isGreaterThanOrEqualTo(VamanaGraphCodec.HEADER_SIZE);

      // Each trailer offset must address that node's degree word: reading the int there equals the
      // node's on-disk degree, and the bytes that follow are its neighbours in order.
      for (int i = 0; i < numNodes; i++) {
        long off = buf.getLong((int) (trailerStart + 8L * i));
        if (i == 0) {
          assertThat(off).isEqualTo(VamanaGraphCodec.HEADER_SIZE);
        }
        int degree = buf.getInt((int) off);
        assertThat(degree).isEqualTo(graph.getNeighbors(i).size());
        for (int k = 0; k < degree; k++) {
          assertThat(buf.getInt((int) (off + 4 + 4L * k))).isEqualTo(graph.getNeighbors(i).node(k));
        }
      }
    }

    @Test
    void rejectsTruncatedOffsetTrailer() {
      byte[] encoded = VamanaGraphCodec.encode(buildSmallGraph(32, 8, 6L));
      // Drop the last 8 bytes (one offset slot) so the trailer is short by one node.
      byte[] truncated = new byte[encoded.length - 8];
      System.arraycopy(encoded, 0, truncated, 0, truncated.length);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(truncated))
          .withMessageContaining("offset trailer");
    }

    @Test
    void rejectsCorruptOffsetInTrailer() {
      byte[] encoded = VamanaGraphCodec.encode(buildSmallGraph(32, 8, 7L));
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      int numNodes = buf.getInt(16);
      // Corrupt offset[1] to point past the body (>= trailer start) → not
      // strictly-increasing/in-body.
      long trailerStart = encoded.length - 8L * numNodes;
      buf.putLong((int) (trailerStart + 8), Long.MAX_VALUE);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("offset");
    }
  }

  // ---------------------------------------------------------------------------
  // Body-validation tests
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class BodyValidation {

    @Test
    void rejectsTruncatedAdjacencyBody() {
      byte[] encoded = VamanaGraphCodec.encode(buildSmallGraph(10, 4, 2L));
      // Chop off everything past the header plus 3 bytes (not a whole int).
      byte[] truncated = new byte[VamanaGraphCodec.HEADER_SIZE + 3];
      System.arraycopy(encoded, 0, truncated, 0, truncated.length);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(truncated))
          .withMessageContaining("truncated");
    }

    @Test
    void rejectsDegreeExceedingMaxDegree() throws IOException {
      // Construct a tiny valid graph, then rewrite the first degree word past maxDegree so decode
      // catches the post-pruning invariant violation.
      VamanaGraph graph = new VamanaGraph(3, 2);
      graph.initNode(0);
      graph.initNode(1);
      graph.initNode(2);
      graph.setMedoid(0);
      graph.getNeighbors(0).insert(1, 10f);
      graph.getNeighbors(0).insert(2, 5f);

      byte[] encoded = VamanaGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // Adjacency body starts at HEADER_SIZE = 28. First word is node 0's degree.
      buf.putInt(VamanaGraphCodec.HEADER_SIZE, 99);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("degree");
    }

    @Test
    void rejectsNeighborOutOfRange() throws IOException {
      VamanaGraph graph = new VamanaGraph(3, 2);
      graph.initNode(0);
      graph.initNode(1);
      graph.initNode(2);
      graph.setMedoid(0);
      graph.getNeighbors(0).insert(1, 10f);
      byte[] encoded = VamanaGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // Adjacency starts at HEADER_SIZE. First entry: degree=1 (4 bytes) then neighbor id (4).
      buf.putInt(VamanaGraphCodec.HEADER_SIZE + 4, 99);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("out of range");
    }

    @Test
    void rejectsSelfLoopNeighbor() throws IOException {
      VamanaGraph graph = new VamanaGraph(3, 2);
      graph.initNode(0);
      graph.initNode(1);
      graph.initNode(2);
      graph.setMedoid(0);
      graph.getNeighbors(0).insert(1, 10f);
      byte[] encoded = VamanaGraphCodec.encode(graph);
      ByteBuffer buf = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      // First neighbor id lives at HEADER_SIZE + 4. Overwrite with 0 (self-loop for node 0).
      buf.putInt(VamanaGraphCodec.HEADER_SIZE + 4, 0);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(encoded))
          .withMessageContaining("self-loop");
    }

    @Test
    void rejectsTrailingBytes() {
      // Extra bytes after a non-empty graph land in the offset-index region, so the trailer-length
      // check (8N bytes expected) rejects them.
      byte[] encoded = VamanaGraphCodec.encode(buildSmallGraph(5, 2, 3L));
      byte[] padded = new byte[encoded.length + 7];
      System.arraycopy(encoded, 0, padded, 0, encoded.length);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(padded))
          .withMessageContaining("offset trailer");
    }

    @Test
    void rejectsEmptyGraphWithTrailingBytes() {
      // An empty graph has a 28-byte header and zero body bytes. Any trailing bytes must be
      // rejected by the empty-graph short circuit, not by the main adjacency loop.
      VamanaGraph empty = new VamanaGraph(1, 16);
      byte[] encoded = VamanaGraphCodec.encode(empty);
      byte[] padded = new byte[encoded.length + 3];
      System.arraycopy(encoded, 0, padded, 0, encoded.length);
      assertThatIOException()
          .isThrownBy(() -> VamanaGraphCodec.decode(padded))
          .withMessageContaining("empty graph");
    }
  }

  // ---------------------------------------------------------------------------
  // Encode-time validation tests (defense-in-depth against corrupt graph state)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class EncodeValidation {

    @Test
    void encodeRejectsSelfLoop() {
      // NeighborArray.insert() does not block inserting node i as its own neighbor, so a buggy
      // or malicious builder could produce a self-loop. encode() must refuse it on the way out
      // so the corruption never reaches graph.bin.
      VamanaGraph graph = new VamanaGraph(3, 2);
      graph.initNode(0);
      graph.initNode(1);
      graph.initNode(2);
      graph.setMedoid(0);
      graph.getNeighbors(0).insert(0, 10f); // self-loop
      assertThatIllegalStateException()
          .isThrownBy(() -> VamanaGraphCodec.encode(graph))
          .withMessageContaining("self-loop");
    }

    @Test
    void encodeRejectsOutOfRangeNeighbor() {
      // NeighborArray.insert() does not bounds-check the nodeId against numNodes, so an
      // off-by-one builder bug could persist a stale ordinal. encode() must catch it before it
      // reaches disk.
      VamanaGraph graph = new VamanaGraph(3, 2);
      graph.initNode(0);
      graph.initNode(1);
      graph.initNode(2);
      graph.setMedoid(0);
      graph.getNeighbors(0).insert(99, 10f); // out of range
      assertThatIllegalStateException()
          .isThrownBy(() -> VamanaGraphCodec.encode(graph))
          .withMessageContaining("out of range");
    }

    @Test
    void encodeRejectsDegreeAboveMaxDegree() {
      // VamanaGraph allocates capacity maxDegree+1 per neighbor array to allow temporary
      // overflow during backlink insertion. By commit time the RobustPruner should have trimmed
      // every list to <= maxDegree. An un-pruned graph (degree > maxDegree) is a builder bug and
      // encode() must fail loudly so the broken state never reaches disk.
      VamanaGraph graph = new VamanaGraph(5, 2);
      for (int i = 0; i < 5; i++) {
        graph.initNode(i);
      }
      graph.setMedoid(0);
      // Fill node 0 to capacity maxDegree+1 = 3 via the overflow slot.
      graph.getNeighbors(0).insert(1, 30f);
      graph.getNeighbors(0).insert(2, 20f);
      graph.getNeighbors(0).insert(3, 10f);
      assertThat(graph.getNeighbors(0).size()).isEqualTo(3);
      assertThatIllegalStateException()
          .isThrownBy(() -> VamanaGraphCodec.encode(graph))
          .withMessageContaining("maxDegree");
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
      assertThatNullPointerException().isThrownBy(() -> VamanaGraphCodec.encode(null));
    }

    @Test
    void decodeRejectsNullBytes() {
      assertThatNullPointerException().isThrownBy(() -> VamanaGraphCodec.decode(null));
    }
  }
}
