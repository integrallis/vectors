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

import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.NeighborArray;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;
import java.util.Objects;

/**
 * Binary codec for the HNSW graph topology file {@code graph.bin}. Every commit on a persistent
 * HNSW collection writes exactly one {@code graph.bin} per generation directory, sitting alongside
 * {@code vectors.bin}, {@code idmap.bin}, and {@code metadata.bin}; every generation open reads it
 * back in one shot and hands the decoded {@link HnswGraph} to {@code MappedHnswIndexAdapter}.
 *
 * <p>Layout (little-endian throughout; format version 1):
 *
 * <pre>
 * Offset  Size   Field                       Notes
 * ------  ----   --------------------------  --------------------------------
 *   0      4     magic                       FileFormat.MAGIC_GRAPH
 *   4      4     format version              FileFormat.VERSION_GRAPH (= 1)
 *   8      4     header length               constant = 32
 *  12      4     flags                       reserved, must be 0
 *  16      4     numNodes                    int32 (must equal manifest.liveCount)
 *  20      4     maxConnections              int32 (HNSW M parameter)
 *  24      4     entryNode                   int32 (-1 iff numNodes == 0)
 *  28      4     maxLevel                    int32 (-1 iff numNodes == 0)
 * ------  ----
 *  32      4*N   nodeLevels[N]               int32 per node, level in [0, maxLevel]
 * ------  ----
 *         varies layer0 adjacency            per node i in [0, numNodes):
 *                                              int32 degree
 *                                              degree * int32 nodeId
 * ------  ----
 *         varies upper-layer adjacency       per layer L in [1, maxLevel]:
 *                                              int32 numNodesAtLayer
 *                                              numNodesAtLayer * {
 *                                                int32 nodeId,
 *                                                int32 degree,
 *                                                degree * int32 neighborNodeId
 *                                              }
 * </pre>
 *
 * <p><b>Why node IDs only, no scores.</b> {@link com.integrallis.vectors.hnsw.HnswSearcher} only
 * reads {@link NeighborArray#node(int)}, never {@link NeighborArray#score(int)} — scores are a
 * builder-only concern. Storing scores would double the file size without any search benefit, and
 * committed decoded graphs are read-only (no incremental insertion, which is when scores would
 * matter).
 *
 * <p><b>How the decoded graph preserves order without scores.</b> {@link NeighborArray#insert}
 * maintains a descending-score sort. On decode, each neighbor ID is inserted with a synthetic score
 * {@code (float) (degree - k)} where {@code k} is the position in the on-disk list. Since we're
 * inserting in strictly decreasing score order, every insert lands at the tail (O(1) arraycopy) and
 * the final iteration order via {@link NeighborArray#node(int)} matches the on-disk order exactly.
 * Scores are unused by the search path so the synthetic values are harmless.
 *
 * <p><b>CRC coverage.</b> {@code graph.bin} is not self-checksummed — its 32-bit CRC lives in the
 * {@linkplain Manifest#graphBinCrc32() manifest}, which is already self-CRC'd. {@code
 * VectorCollectionImpl.openGeneration} verifies the graph CRC before calling {@link #decode} so a
 * corrupt or truncated file reaches the decoder only if the manifest is also corrupt (in which case
 * the generation is discarded and recovery walks backward).
 */
public final class HnswGraphCodec {

  /** Fixed header length in bytes, including magic/version/flags/counts. */
  public static final int HEADER_SIZE = 32;

  /**
   * Sanity upper bound on an encoded graph. Anything larger would overflow a {@code byte[]}
   * allocation (Java caps arrays at just under 2 GB). At M=32 and the M-dominated size model of
   * {@code N * (2M+1) * 4}, this corresponds to roughly 8 million nodes — well above the target
   * workload for {@code vectors-db}.
   */
  private static final long MAX_ENCODED_SIZE = Integer.MAX_VALUE - 8L;

  private HnswGraphCodec() {}

  /**
   * Serializes {@code graph} into a fresh byte array that matches the file-format spec in this
   * class's Javadoc. The result includes the 32-byte header, the per-node level vector, the full
   * layer-0 adjacency lists, and every upper-layer adjacency list.
   *
   * @throws NullPointerException if {@code graph} is null
   * @throws IllegalStateException if the graph is in an inconsistent state (e.g. a live node
   *     missing its layer-0 neighbor list) or if the resulting byte count would exceed {@code 2 GB}
   */
  public static byte[] encode(HnswGraph graph) {
    Objects.requireNonNull(graph, "graph must not be null");

    int numNodes = graph.size();
    int maxConnections = graph.maxConnections();
    int entryNode = graph.entryNode();
    int maxLevel = graph.maxLevel();

    if (numNodes < 0) {
      throw new IllegalStateException("graph.size() is negative: " + numNodes);
    }
    if (maxConnections <= 0) {
      throw new IllegalStateException("graph.maxConnections() must be positive: " + maxConnections);
    }

    // Pre-compute the encoded size in one pass to avoid a growing buffer. Also doubles as an
    // invariant check: if any live node has a null layer-0 list, we throw here rather than
    // writing a half-formed file.
    long totalBytes = HEADER_SIZE;
    totalBytes += 4L * numNodes; // nodeLevels

    // Layer 0 sizing.
    for (int i = 0; i < numNodes; i++) {
      NeighborArray layer0 = graph.getNeighbors(i, 0);
      if (layer0 == null) {
        throw new IllegalStateException(
            "layer 0 neighbor list missing for live node " + i + " (graph is corrupt)");
      }
      totalBytes += 4L + 4L * layer0.size(); // degree + neighbors
    }

    // Upper-layer sizing. Cache counts so the write pass doesn't recompute.
    int[] upperLayerCounts = new int[Math.max(0, maxLevel)];
    for (int layer = 1; layer <= maxLevel; layer++) {
      int count = 0;
      long edgeBytes = 0;
      for (int i = 0; i < numNodes; i++) {
        NeighborArray neighbors = graph.getNeighbors(i, layer);
        if (neighbors != null) {
          count++;
          edgeBytes += 4L + 4L + 4L * neighbors.size(); // nodeId + degree + neighbors
        }
      }
      upperLayerCounts[layer - 1] = count;
      totalBytes += 4L + edgeBytes; // numNodesAtLayer + body
    }

    if (totalBytes > MAX_ENCODED_SIZE) {
      throw new IllegalStateException(
          "HnswGraph exceeds the 2 GB graph.bin limit: " + totalBytes + " bytes");
    }

    byte[] out = new byte[(int) totalBytes];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    // --- Header ------------------------------------------------------------
    buf.putInt(FileFormat.MAGIC_GRAPH);
    buf.putInt(FileFormat.VERSION_GRAPH);
    buf.putInt(HEADER_SIZE);
    buf.putInt(0); // flags reserved
    buf.putInt(numNodes);
    buf.putInt(maxConnections);
    buf.putInt(entryNode);
    buf.putInt(maxLevel);

    // --- nodeLevels --------------------------------------------------------
    for (int i = 0; i < numNodes; i++) {
      buf.putInt(graph.nodeLevel(i));
    }

    // Scratch for duplicate-neighbor detection. A single reusable BitSet per encode call;
    // cleared before each neighbor list so we catch duplicates *within* a node's adjacency
    // list at write time rather than leaving them for decode() to reject.
    BitSet seen = new BitSet(numNodes);

    // --- Layer 0 adjacency -------------------------------------------------
    for (int i = 0; i < numNodes; i++) {
      NeighborArray layer0 = graph.getNeighbors(i, 0);
      int degree = layer0.size();
      buf.putInt(degree);
      seen.clear();
      for (int k = 0; k < degree; k++) {
        int neighborId = layer0.node(k);
        if (neighborId < 0 || neighborId >= numNodes) {
          throw new IllegalStateException(
              "layer 0 neighbor " + neighborId + " for node " + i + " out of range");
        }
        if (neighborId == i) {
          throw new IllegalStateException("layer 0 self-loop on node " + i);
        }
        if (seen.get(neighborId)) {
          throw new IllegalStateException(
              "layer 0 duplicate neighbor " + neighborId + " in node " + i);
        }
        seen.set(neighborId);
        buf.putInt(neighborId);
      }
    }

    // --- Upper-layer adjacency ---------------------------------------------
    for (int layer = 1; layer <= maxLevel; layer++) {
      buf.putInt(upperLayerCounts[layer - 1]);
      for (int i = 0; i < numNodes; i++) {
        NeighborArray neighbors = graph.getNeighbors(i, layer);
        if (neighbors == null) {
          continue;
        }
        buf.putInt(i);
        int degree = neighbors.size();
        buf.putInt(degree);
        seen.clear();
        for (int k = 0; k < degree; k++) {
          int neighborId = neighbors.node(k);
          if (neighborId < 0 || neighborId >= numNodes) {
            throw new IllegalStateException(
                "layer " + layer + " neighbor " + neighborId + " for node " + i + " out of range");
          }
          if (neighborId == i) {
            throw new IllegalStateException("layer " + layer + " self-loop on node " + i);
          }
          if (seen.get(neighborId)) {
            throw new IllegalStateException(
                "layer " + layer + " duplicate neighbor " + neighborId + " in node " + i);
          }
          seen.set(neighborId);
          buf.putInt(neighborId);
        }
      }
    }

    return out;
  }

  /**
   * Parses the given bytes as a {@code graph.bin} image and returns a fully-populated {@link
   * HnswGraph}. Every invariant from the format spec is checked; any deviation throws {@link
   * IOException} with a message identifying the first failing invariant.
   *
   * @throws NullPointerException if {@code bytes} is null
   * @throws IOException if the bytes are truncated, wrong magic/version/flags, or encode an
   *     inconsistent graph (out-of-range node ID, degree larger than capacity, duplicate neighbor,
   *     node level exceeding the declared max level, etc.)
   */
  public static HnswGraph decode(byte[] bytes) throws IOException {
    Objects.requireNonNull(bytes, "bytes must not be null");
    if (bytes.length < HEADER_SIZE) {
      throw new IOException(
          "graph.bin truncated: expected at least "
              + HEADER_SIZE
              + " header bytes, got "
              + bytes.length);
    }

    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

    int magic = buf.getInt();
    if (magic != FileFormat.MAGIC_GRAPH) {
      throw new IOException(
          String.format(
              "graph.bin magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_GRAPH, magic));
    }
    int version = buf.getInt();
    if (version != FileFormat.VERSION_GRAPH) {
      throw new IOException(
          "graph.bin version mismatch: expected " + FileFormat.VERSION_GRAPH + ", got " + version);
    }
    int headerLength = buf.getInt();
    if (headerLength != HEADER_SIZE) {
      throw new IOException(
          "graph.bin header length mismatch: expected " + HEADER_SIZE + ", got " + headerLength);
    }
    int flags = buf.getInt();
    if (flags != 0) {
      throw new IOException("graph.bin flags must be 0 in version 1, got " + flags);
    }

    int numNodes = buf.getInt();
    if (numNodes < 0) {
      throw new IOException("graph.bin numNodes must be >= 0, got " + numNodes);
    }
    int maxConnections = buf.getInt();
    if (maxConnections <= 0) {
      throw new IOException("graph.bin maxConnections must be positive, got " + maxConnections);
    }
    int entryNode = buf.getInt();
    int maxLevel = buf.getInt();

    // Empty-graph short circuit. We still return a valid, zero-node HnswGraph; the constructor
    // insists on maxNodes > 0, so we pass 1 as a sentinel — the graph's size() stays 0.
    if (numNodes == 0) {
      if (entryNode != -1 || maxLevel != -1) {
        throw new IOException(
            "graph.bin empty graph must have entryNode=-1 and maxLevel=-1, got entryNode="
                + entryNode
                + ", maxLevel="
                + maxLevel);
      }
      if (buf.remaining() != 0) {
        throw new IOException(
            "graph.bin empty graph must have no body, got " + buf.remaining() + " trailing bytes");
      }
      return new HnswGraph(1, maxConnections);
    }

    // Non-empty graph validation.
    if (entryNode < 0 || entryNode >= numNodes) {
      throw new IOException(
          "graph.bin entryNode " + entryNode + " out of range [0, " + numNodes + ")");
    }
    if (maxLevel < 0) {
      throw new IOException("graph.bin maxLevel must be >= 0 when numNodes > 0, got " + maxLevel);
    }

    // Widen buf.remaining() to long via the 4L literal so the arithmetic cannot overflow even at
    // the theoretical numNodes > 536M ceiling that would wrap a naive 4 * numNodes computation.
    // The MAX_ENCODED_SIZE guard in encode() keeps numNodes well below that bound in practice.
    if (buf.remaining() < 4L * numNodes) {
      throw new IOException(
          "graph.bin truncated nodeLevels section: need "
              + (4L * numNodes)
              + " bytes, have "
              + buf.remaining());
    }
    int[] nodeLevels = new int[numNodes];
    for (int i = 0; i < numNodes; i++) {
      int level = buf.getInt();
      if (level < 0 || level > maxLevel) {
        throw new IOException(
            "graph.bin nodeLevels[" + i + "]=" + level + " out of range [0, " + maxLevel + "]");
      }
      nodeLevels[i] = level;
    }

    // HNSW invariant: the entry node must live at the highest level. A graph where
    // nodeLevels[entryNode] < maxLevel would decode without error but silently produce wrong
    // search results (greedy descend would start at a lower layer and miss the top-layer
    // connections needed to reach the correct neighborhood). Catch it here.
    if (nodeLevels[entryNode] != maxLevel) {
      throw new IOException(
          "graph.bin entryNode "
              + entryNode
              + " lives at level "
              + nodeLevels[entryNode]
              + " but maxLevel is "
              + maxLevel);
    }

    // Construct the graph and initialize every node at the correct level.
    HnswGraph graph = new HnswGraph(numNodes, maxConnections);
    for (int i = 0; i < numNodes; i++) {
      graph.initNode(i, nodeLevels[i]);
    }
    graph.setEntryNode(entryNode, maxLevel);

    int maxConnections0 = 2 * maxConnections;

    // --- Layer 0 adjacency -------------------------------------------------
    for (int i = 0; i < numNodes; i++) {
      int degree = readNonNegativeInt(buf, "layer 0 degree for node " + i);
      if (degree > maxConnections0) {
        throw new IOException(
            "graph.bin layer 0 degree "
                + degree
                + " for node "
                + i
                + " exceeds 2*M = "
                + maxConnections0);
      }
      NeighborArray arr = graph.getNeighbors(i, 0);
      if (arr == null) {
        // Unreachable if initNode ran — kept as a defensive invariant.
        throw new IOException("graph.bin layer 0 slot missing for node " + i);
      }
      loadNeighborsInOrder(buf, arr, degree, numNodes, i, 0);
    }

    // --- Upper-layer adjacency ---------------------------------------------
    for (int layer = 1; layer <= maxLevel; layer++) {
      int numNodesAtLayer = readNonNegativeInt(buf, "numNodesAtLayer for layer " + layer);
      if (numNodesAtLayer > numNodes) {
        throw new IOException(
            "graph.bin numNodesAtLayer "
                + numNodesAtLayer
                + " for layer "
                + layer
                + " exceeds numNodes "
                + numNodes);
      }
      int lastNodeId = -1;
      for (int j = 0; j < numNodesAtLayer; j++) {
        int nodeId = readNonNegativeInt(buf, "upper-layer nodeId at layer " + layer);
        if (nodeId >= numNodes) {
          throw new IOException(
              "graph.bin layer "
                  + layer
                  + " references nodeId "
                  + nodeId
                  + " out of range [0, "
                  + numNodes
                  + ")");
        }
        if (nodeId <= lastNodeId) {
          // The encoder emits nodes in ascending ID order; any violation indicates corruption.
          throw new IOException(
              "graph.bin layer "
                  + layer
                  + " nodeIds must be strictly increasing, got "
                  + nodeId
                  + " after "
                  + lastNodeId);
        }
        lastNodeId = nodeId;
        if (nodeLevels[nodeId] < layer) {
          throw new IOException(
              "graph.bin layer "
                  + layer
                  + " contains node "
                  + nodeId
                  + " whose nodeLevels entry is only "
                  + nodeLevels[nodeId]);
        }
        int degree = readNonNegativeInt(buf, "upper-layer degree for node " + nodeId);
        if (degree > maxConnections) {
          throw new IOException(
              "graph.bin layer "
                  + layer
                  + " degree "
                  + degree
                  + " for node "
                  + nodeId
                  + " exceeds M = "
                  + maxConnections);
        }
        NeighborArray arr = graph.getNeighbors(nodeId, layer);
        if (arr == null) {
          throw new IOException(
              "graph.bin layer "
                  + layer
                  + " slot missing for node "
                  + nodeId
                  + " (nodeLevels mismatch)");
        }
        loadNeighborsInOrder(buf, arr, degree, numNodes, nodeId, layer);
      }
    }

    if (buf.remaining() != 0) {
      throw new IOException(
          "graph.bin has " + buf.remaining() + " trailing bytes after full decode");
    }

    return graph;
  }

  /**
   * Reads {@code degree} neighbor ids from {@code buf} and inserts them into {@code dst} while
   * preserving the on-disk order. Uses synthetic monotonically decreasing scores so that {@link
   * NeighborArray#insert} always places each new id at the tail (O(1) arraycopy), and the final
   * iteration order matches the bytes on disk.
   */
  private static void loadNeighborsInOrder(
      ByteBuffer buf, NeighborArray dst, int degree, int numNodes, int ownerNode, int layer)
      throws IOException {
    if (buf.remaining() < 4L * degree) {
      throw new IOException(
          "graph.bin truncated at layer "
              + layer
              + " node "
              + ownerNode
              + " neighbor list (need "
              + (4L * degree)
              + " bytes, have "
              + buf.remaining()
              + ")");
    }
    for (int k = 0; k < degree; k++) {
      int neighborId = buf.getInt();
      if (neighborId < 0 || neighborId >= numNodes) {
        throw new IOException(
            "graph.bin layer "
                + layer
                + " node "
                + ownerNode
                + " neighbor "
                + neighborId
                + " out of range [0, "
                + numNodes
                + ")");
      }
      if (neighborId == ownerNode) {
        throw new IOException(
            "graph.bin layer " + layer + " node " + ownerNode + " contains a self-loop");
      }
      // Synthetic score: first neighbor gets the highest score, last gets 1. Guaranteed positive
      // so the NodeQueue-style encode() guards downstream never see a negative score.
      float syntheticScore = (float) (degree - k);
      if (!dst.insert(neighborId, syntheticScore)) {
        throw new IOException(
            "graph.bin layer "
                + layer
                + " node "
                + ownerNode
                + " contains duplicate neighbor "
                + neighborId);
      }
    }
  }

  private static int readNonNegativeInt(ByteBuffer buf, String what) throws IOException {
    if (buf.remaining() < 4) {
      throw new IOException("graph.bin truncated while reading " + what);
    }
    int value = buf.getInt();
    if (value < 0) {
      throw new IOException("graph.bin " + what + " must be >= 0, got " + value);
    }
    return value;
  }
}
