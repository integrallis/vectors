package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.vamana.NeighborArray;
import com.integrallis.vectors.vamana.VamanaGraph;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;
import java.util.Objects;

/**
 * Binary codec for the Vamana graph topology file {@code graph.bin}. Every commit on a persistent
 * Vamana collection writes exactly one {@code graph.bin} per generation directory, sitting
 * alongside {@code vectors.bin}, {@code idmap.bin}, and {@code metadata.bin}; every generation open
 * reads it back in one shot and hands the decoded {@link VamanaGraph} to {@code
 * MappedVamanaIndexAdapter}.
 *
 * <p>Vamana's single-layer flat graph is substantially simpler than HNSW's multi-layer structure:
 * there are no per-node levels, no upper-layer adjacency sections, and the entry point is a single
 * {@code medoid} integer. The entire body is a straight sequence of {@code (degree, neighbors...)}
 * tuples, one per node.
 *
 * <p>Layout (little-endian throughout; Step 4c version 1):
 *
 * <pre>
 * Offset  Size   Field                       Notes
 * ------  ----   --------------------------  --------------------------------
 *   0      4     magic                       FileFormat.MAGIC_GRAPH_VAMANA
 *   4      4     format version              FileFormat.VERSION_GRAPH_VAMANA (= 1)
 *   8      4     header length               constant = 28
 *  12      4     flags                       reserved, must be 0
 *  16      4     numNodes                    int32 (must equal manifest.liveCount)
 *  20      4     maxDegree                   int32 (Vamana R parameter)
 *  24      4     medoid                      int32 (-1 iff numNodes == 0; else in [0, numNodes))
 * ------  ----
 *  28      varies adjacency                  per node i in [0, numNodes):
 *                                              int32 degree
 *                                              degree * int32 nodeId
 * </pre>
 *
 * <p><b>Why node IDs only, no scores.</b> {@link com.integrallis.vectors.vamana.VamanaSearcher}
 * only reads {@link NeighborArray#node(int)}, never {@link NeighborArray#score(int)} — scores are a
 * builder-only concern (used by the robust pruner to measure diversity during construction).
 * Storing scores would nearly double the file size without any search benefit, and Step 4c uses
 * rebuild-on-commit so the decoded graph is read-only.
 *
 * <p><b>How the decoded graph preserves order without scores.</b> {@link NeighborArray#insert}
 * maintains a descending-score sort. On decode, each neighbor ID is inserted with a synthetic score
 * {@code (float) (degree - k)} where {@code k} is the position in the on-disk list. Since we insert
 * in strictly decreasing score order, every insert lands at the tail (O(1) arraycopy) and the final
 * iteration order via {@link NeighborArray#node(int)} matches the on-disk order exactly. Scores are
 * unused by the search path so the synthetic values are harmless.
 *
 * <p><b>CRC coverage.</b> {@code graph.bin} is not self-checksummed — its 32-bit CRC lives in the
 * {@linkplain Manifest#graphBinCrc32() manifest}, which is already self-CRC'd. {@code
 * VectorCollectionImpl.openGeneration} verifies the graph CRC before calling {@link #decode} so a
 * corrupt or truncated file reaches the decoder only if the manifest is also corrupt (in which case
 * the generation is discarded and recovery walks backward).
 *
 * <p><b>Why a distinct magic from {@link HnswGraphCodec}.</b> HNSW and Vamana share the same
 * filename {@code graph.bin} and the same manifest slot ({@code graphBinLength}/{@code
 * graphBinCrc32}) but use incompatible on-disk layouts. Using {@link FileFormat#MAGIC_GRAPH_VAMANA}
 * means a mistakenly-dispatched decoder (an HNSW decode against a Vamana file, or vice versa) fails
 * at the first 4-byte read rather than silently producing a wrong-shaped graph.
 */
public final class VamanaGraphCodec {

  /** Fixed header length in bytes, including magic/version/flags/counts. */
  public static final int HEADER_SIZE = 28;

  /**
   * Sanity upper bound on an encoded graph. Anything larger would overflow a {@code byte[]}
   * allocation (Java caps arrays at just under 2 GB). At R=64 and the R-dominated size model of
   * {@code N * (R+1) * 4}, this corresponds to roughly 8 million nodes — well above the target
   * workload for {@code vectors-db}.
   */
  private static final long MAX_ENCODED_SIZE = Integer.MAX_VALUE - 8L;

  private VamanaGraphCodec() {}

  /**
   * Serializes {@code graph} into a fresh byte array that matches the file-format spec in this
   * class's Javadoc. The result includes the 28-byte header and the flat adjacency body.
   *
   * @throws NullPointerException if {@code graph} is null
   * @throws IllegalStateException if the graph is in an inconsistent state (e.g. a degree above
   *     {@code maxDegree}, a self-loop, a duplicate neighbor, a medoid out of range) or if the
   *     resulting byte count would exceed {@code 2 GB}
   */
  public static byte[] encode(VamanaGraph graph) {
    Objects.requireNonNull(graph, "graph must not be null");

    int numNodes = graph.size();
    int maxDegree = graph.maxDegree();

    if (numNodes < 0) {
      throw new IllegalStateException("graph.size() is negative: " + numNodes);
    }
    if (maxDegree <= 0) {
      throw new IllegalStateException("graph.maxDegree() must be positive: " + maxDegree);
    }

    // Pre-compute the encoded size in one pass. Also doubles as an invariant check: if any live
    // node's adjacency list is over-capacity or contains a self-loop or a duplicate, we throw
    // here rather than writing a half-formed file.
    long totalBytes = HEADER_SIZE;
    for (int i = 0; i < numNodes; i++) {
      NeighborArray neighbors = graph.getNeighbors(i);
      int degree = neighbors.size();
      if (degree > maxDegree) {
        throw new IllegalStateException(
            "node "
                + i
                + " has degree "
                + degree
                + " > maxDegree "
                + maxDegree
                + " (graph is un-pruned)");
      }
      totalBytes += 4L + 4L * degree; // degree word + neighbor IDs
    }

    if (totalBytes > MAX_ENCODED_SIZE) {
      throw new IllegalStateException(
          "VamanaGraph exceeds the 2 GB graph.bin limit: " + totalBytes + " bytes");
    }

    // Medoid validation. An empty graph has no meaningful medoid; we emit -1 as a sentinel
    // matching HnswGraphCodec's entryNode=-1 pattern. A non-empty graph must have a medoid in
    // range — VamanaGraph's setMedoid guards this at insert time but we re-check here so the
    // file's on-disk invariant is explicit.
    int medoid;
    if (numNodes == 0) {
      medoid = -1;
    } else {
      medoid = graph.medoid();
      if (medoid < 0 || medoid >= numNodes) {
        throw new IllegalStateException(
            "graph.medoid() " + medoid + " out of range [0, " + numNodes + ")");
      }
    }

    byte[] out = new byte[(int) totalBytes];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);

    // --- Header ------------------------------------------------------------
    buf.putInt(FileFormat.MAGIC_GRAPH_VAMANA);
    buf.putInt(FileFormat.VERSION_GRAPH_VAMANA);
    buf.putInt(HEADER_SIZE);
    buf.putInt(0); // flags reserved
    buf.putInt(numNodes);
    buf.putInt(maxDegree);
    buf.putInt(medoid);

    // Scratch for duplicate-neighbor detection. A single reusable BitSet per encode call;
    // cleared before each neighbor list so we catch duplicates *within* a node's adjacency list
    // at write time rather than leaving them for decode() to reject.
    BitSet seen = new BitSet(numNodes);

    // --- Flat adjacency body ----------------------------------------------
    for (int i = 0; i < numNodes; i++) {
      NeighborArray neighbors = graph.getNeighbors(i);
      int degree = neighbors.size();
      buf.putInt(degree);
      seen.clear();
      for (int k = 0; k < degree; k++) {
        int neighborId = neighbors.node(k);
        if (neighborId < 0 || neighborId >= numNodes) {
          throw new IllegalStateException(
              "neighbor " + neighborId + " for node " + i + " out of range [0, " + numNodes + ")");
        }
        if (neighborId == i) {
          throw new IllegalStateException("self-loop on node " + i);
        }
        if (seen.get(neighborId)) {
          throw new IllegalStateException("duplicate neighbor " + neighborId + " in node " + i);
        }
        seen.set(neighborId);
        buf.putInt(neighborId);
      }
    }

    return out;
  }

  /**
   * Parses the given bytes as a Vamana {@code graph.bin} image and returns a fully-populated {@link
   * VamanaGraph}. Every invariant from the format spec is checked; any deviation throws {@link
   * IOException} with a message identifying the first failing invariant.
   *
   * @throws NullPointerException if {@code bytes} is null
   * @throws IOException if the bytes are truncated, wrong magic/version/flags, or encode an
   *     inconsistent graph (out-of-range node ID, degree larger than maxDegree, duplicate neighbor,
   *     self-loop, medoid out of range, etc.)
   */
  public static VamanaGraph decode(byte[] bytes) throws IOException {
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
    if (magic != FileFormat.MAGIC_GRAPH_VAMANA) {
      throw new IOException(
          String.format(
              "graph.bin (vamana) magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_GRAPH_VAMANA, magic));
    }
    int version = buf.getInt();
    if (version != FileFormat.VERSION_GRAPH_VAMANA) {
      throw new IOException(
          "graph.bin (vamana) version mismatch: expected "
              + FileFormat.VERSION_GRAPH_VAMANA
              + ", got "
              + version);
    }
    int headerLength = buf.getInt();
    if (headerLength != HEADER_SIZE) {
      throw new IOException(
          "graph.bin (vamana) header length mismatch: expected "
              + HEADER_SIZE
              + ", got "
              + headerLength);
    }
    int flags = buf.getInt();
    if (flags != 0) {
      throw new IOException("graph.bin (vamana) flags must be 0 in version 1, got " + flags);
    }

    int numNodes = buf.getInt();
    if (numNodes < 0) {
      throw new IOException("graph.bin (vamana) numNodes must be >= 0, got " + numNodes);
    }
    int maxDegree = buf.getInt();
    if (maxDegree <= 0) {
      throw new IOException("graph.bin (vamana) maxDegree must be positive, got " + maxDegree);
    }
    int medoid = buf.getInt();

    // Empty-graph short circuit. We still return a valid, zero-node VamanaGraph; the constructor
    // insists on maxNodes > 0, so we pass 1 as a sentinel — the graph's size() stays 0 because
    // we do not call initNode.
    if (numNodes == 0) {
      if (medoid != -1) {
        throw new IOException(
            "graph.bin (vamana) empty graph must have medoid=-1, got medoid=" + medoid);
      }
      if (buf.remaining() != 0) {
        throw new IOException(
            "graph.bin (vamana) empty graph must have no body, got "
                + buf.remaining()
                + " trailing bytes");
      }
      return new VamanaGraph(1, maxDegree);
    }

    // Non-empty graph validation.
    if (medoid < 0 || medoid >= numNodes) {
      throw new IOException(
          "graph.bin (vamana) medoid " + medoid + " out of range [0, " + numNodes + ")");
    }

    // Construct the graph and initialize every node. VamanaGraph.initNode allocates a
    // NeighborArray of capacity maxDegree+1 per node.
    VamanaGraph graph = new VamanaGraph(numNodes, maxDegree);
    for (int i = 0; i < numNodes; i++) {
      graph.initNode(i);
    }
    graph.setMedoid(medoid);

    // --- Flat adjacency body ----------------------------------------------
    for (int i = 0; i < numNodes; i++) {
      int degree = readNonNegativeInt(buf, "degree for node " + i);
      if (degree > maxDegree) {
        throw new IOException(
            "graph.bin (vamana) degree "
                + degree
                + " for node "
                + i
                + " exceeds maxDegree "
                + maxDegree);
      }
      NeighborArray arr = graph.getNeighbors(i);
      loadNeighborsInOrder(buf, arr, degree, numNodes, i);
    }

    if (buf.remaining() != 0) {
      throw new IOException(
          "graph.bin (vamana) has " + buf.remaining() + " trailing bytes after full decode");
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
      ByteBuffer buf, NeighborArray dst, int degree, int numNodes, int ownerNode)
      throws IOException {
    if (buf.remaining() < 4L * degree) {
      throw new IOException(
          "graph.bin (vamana) truncated at node "
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
            "graph.bin (vamana) node "
                + ownerNode
                + " neighbor "
                + neighborId
                + " out of range [0, "
                + numNodes
                + ")");
      }
      if (neighborId == ownerNode) {
        throw new IOException("graph.bin (vamana) node " + ownerNode + " contains a self-loop");
      }
      // Synthetic score: first neighbor gets the highest score, last gets 1. Guaranteed positive
      // so downstream NodeQueue-style encoders that forbid negative scores stay happy.
      float syntheticScore = (float) (degree - k);
      if (!dst.insert(neighborId, syntheticScore)) {
        throw new IOException(
            "graph.bin (vamana) node " + ownerNode + " contains duplicate neighbor " + neighborId);
      }
    }
  }

  private static int readNonNegativeInt(ByteBuffer buf, String what) throws IOException {
    if (buf.remaining() < 4) {
      throw new IOException("graph.bin (vamana) truncated while reading " + what);
    }
    int value = buf.getInt();
    if (value < 0) {
      throw new IOException("graph.bin (vamana) " + what + " must be >= 0, got " + value);
    }
    return value;
  }
}
