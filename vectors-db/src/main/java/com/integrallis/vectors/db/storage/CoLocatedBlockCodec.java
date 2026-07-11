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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Co-located block format for the object-storage vector index (SOTA design P2).
 *
 * <p>Each graph node occupies one <em>fixed-stride block</em> holding its own quantization code,
 * its adjacency, and — co-located, SymphonyQG-style — <em>every neighbor's code</em>. Expanding a
 * node in the beam therefore reads exactly one block (one ranged object-store GET) and can score
 * the whole frontier from that block with zero further reads. Full-precision vectors live in a
 * separate blob and are fetched (ranged GET) only for the small rerank candidate set.
 *
 * <p>Fixed stride gives DiskANN-style O(1) addressing — {@code offset(node) = HEADER_SIZE + node *
 * stride} — with no trailing offset index. Each block is self-describing via a leading degree word;
 * unused neighbor slots (degree &lt; maxDegree) are left as zero padding to the stride.
 *
 * <pre>
 * Header (36 bytes, little-endian):
 *   0   4  magic       0x56434C42 ("VCLB")
 *   4   4  version     1
 *   8   4  headerLen   36
 *   12  4  flags       reserved (0)
 *   16  4  numNodes
 *   20  4  maxDegree   per-block neighbor capacity (e.g. layer-0 2*M)
 *   24  4  codeBytes   quantization code length per vector
 *   28  4  stride      bytes per node block
 *   32  4  entryNode   search entry point (-1 if empty)
 * Body: numNodes blocks, each `stride` bytes:
 *   int32 degree
 *   byte[codeBytes]  this node's code
 *   { int32 neighborId ; byte[codeBytes] neighborCode } * maxDegree   (first `degree` valid, rest zero)
 * </pre>
 */
public final class CoLocatedBlockCodec {

  static final int MAGIC = 0x56434C42; // "VCLB"
  static final int VERSION = 1;
  static final int HEADER_SIZE = 36;

  private static final ValueLayout.OfInt LE_INT =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private CoLocatedBlockCodec() {}

  /** Bytes each node block occupies for the given degree capacity and code length. */
  public static int strideFor(int maxDegree, int codeBytes) {
    int nbrEntry = Integer.BYTES + codeBytes; // neighborId + neighborCode
    return Integer.BYTES /* degree */ + codeBytes /* node code */ + maxDegree * nbrEntry;
  }

  /**
   * Encodes the co-located block file.
   *
   * @param numNodes number of nodes (== live count)
   * @param maxDegree per-block neighbor capacity (typically the layer-0 max, {@code 2*M})
   * @param codeBytes quantization code length per vector
   * @param entryNode search entry point, or -1 if empty
   * @param adjacency per node, its neighbor ids (length may exceed maxDegree — truncated)
   * @param codes per node, its {@code codeBytes}-long quantization code
   * @return the encoded blob
   */
  public static byte[] encode(
      int numNodes,
      int maxDegree,
      int codeBytes,
      int entryNode,
      int[][] adjacency,
      byte[][] codes) {
    if (adjacency.length != numNodes || codes.length != numNodes) {
      throw new IllegalArgumentException("adjacency/codes length must equal numNodes");
    }
    int stride = strideFor(maxDegree, codeBytes);
    long total = (long) HEADER_SIZE + (long) numNodes * stride;
    if (total > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("blob exceeds max array length: " + total);
    }
    ByteBuffer buf = ByteBuffer.allocate((int) total).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(MAGIC).putInt(VERSION).putInt(HEADER_SIZE).putInt(0);
    buf.putInt(numNodes).putInt(maxDegree).putInt(codeBytes).putInt(stride).putInt(entryNode);

    int nbrEntry = Integer.BYTES + codeBytes;
    for (int node = 0; node < numNodes; node++) {
      byte[] nodeCode = codes[node];
      if (nodeCode.length != codeBytes) {
        throw new IllegalArgumentException(
            "code[" + node + "] length " + nodeCode.length + " != codeBytes " + codeBytes);
      }
      int base = HEADER_SIZE + node * stride;
      int deg = Math.min(adjacency[node].length, maxDegree);
      buf.position(base);
      buf.putInt(deg);
      buf.put(nodeCode);
      int slot = base + Integer.BYTES + codeBytes;
      for (int i = 0; i < deg; i++) {
        int nbr = adjacency[node][i];
        buf.position(slot);
        buf.putInt(nbr);
        buf.put(codes[nbr]);
        slot += nbrEntry;
      }
      // remaining neighbor slots stay zero (ByteBuffer.allocate zero-fills)
    }
    return buf.array();
  }

  /** Opens a zero-copy paged reader over the encoded blob (mmap or ranged-GET backed). */
  public static Reader open(MemorySegment seg) {
    int magic = seg.get(LE_INT, 0);
    if (magic != MAGIC) {
      throw new IllegalArgumentException(String.format("bad magic 0x%08X", magic));
    }
    int version = seg.get(LE_INT, 4);
    if (version != VERSION) {
      throw new IllegalArgumentException("unsupported version " + version);
    }
    int headerLen = seg.get(LE_INT, 8);
    if (headerLen != HEADER_SIZE) {
      throw new IllegalArgumentException("bad header length " + headerLen);
    }
    int numNodes = seg.get(LE_INT, 16);
    int maxDegree = seg.get(LE_INT, 20);
    int codeBytes = seg.get(LE_INT, 24);
    int stride = seg.get(LE_INT, 28);
    int entryNode = seg.get(LE_INT, 32);
    if (stride != strideFor(maxDegree, codeBytes)) {
      throw new IllegalArgumentException("stride mismatch");
    }
    long expected = (long) HEADER_SIZE + (long) numNodes * stride;
    if (seg.byteSize() < expected) {
      throw new IllegalArgumentException("segment too small: " + seg.byteSize() + " < " + expected);
    }
    return new Reader(seg, numNodes, maxDegree, codeBytes, stride, entryNode);
  }

  /**
   * Zero-copy paged view over the block file. A node's block is {@code [off(node),
   * off(node)+stride)} — exactly the byte range one ranged GET fetches. All accessors are O(1)
   * offset arithmetic.
   */
  public static final class Reader {
    private final MemorySegment seg;
    private final int numNodes;
    private final int maxDegree;
    private final int codeBytes;
    private final int stride;
    private final int entryNode;
    private final int nbrEntry;

    private Reader(
        MemorySegment seg, int numNodes, int maxDegree, int codeBytes, int stride, int entryNode) {
      this.seg = seg;
      this.numNodes = numNodes;
      this.maxDegree = maxDegree;
      this.codeBytes = codeBytes;
      this.stride = stride;
      this.entryNode = entryNode;
      this.nbrEntry = Integer.BYTES + codeBytes;
    }

    public int numNodes() {
      return numNodes;
    }

    public int maxDegree() {
      return maxDegree;
    }

    public int codeBytes() {
      return codeBytes;
    }

    public int stride() {
      return stride;
    }

    public int entryNode() {
      return entryNode;
    }

    /** Absolute byte offset of a node's block (== the ranged-GET start). */
    public long blockOffset(int node) {
      return (long) HEADER_SIZE + (long) node * stride;
    }

    public int degree(int node) {
      return seg.get(LE_INT, blockOffset(node));
    }

    /** Copies a node's neighbor ids into {@code out}; returns the degree. */
    public int neighbors(int node, int[] out) {
      int deg = degree(node);
      long slot = blockOffset(node) + Integer.BYTES + codeBytes;
      for (int i = 0; i < deg; i++) {
        out[i] = seg.get(LE_INT, slot);
        slot += nbrEntry;
      }
      return deg;
    }

    /** Zero-copy slice of a node's own quantization code. */
    public MemorySegment nodeCode(int node) {
      return seg.asSlice(blockOffset(node) + Integer.BYTES, codeBytes);
    }

    /** Zero-copy slice of the code of a node's {@code i}-th neighbor (co-located in the block). */
    public MemorySegment neighborCode(int node, int i) {
      long o = blockOffset(node) + Integer.BYTES + codeBytes + (long) i * nbrEntry + Integer.BYTES;
      return seg.asSlice(o, codeBytes);
    }
  }
}
