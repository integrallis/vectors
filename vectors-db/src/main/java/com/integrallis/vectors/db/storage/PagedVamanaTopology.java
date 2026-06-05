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

import static java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED;

import com.integrallis.vectors.vamana.VamanaTopology;
import java.io.IOException;
import java.lang.foreign.MemorySegment;

/**
 * Disk-resident {@link VamanaTopology} backed by an mmap'd {@code graph.bin} (format v2). Instead
 * of decoding the whole adjacency into heap, it pages each node's neighbour list straight from the
 * {@link MemorySegment} on demand, using the trailing per-node offset index for O(1) addressing
 * (I.4). The graph adjacency therefore lives in the OS page cache, not the Java heap — search heap
 * drops from O(N·R) to O(threads·L).
 *
 * <p>The backing segment is owned by the caller (the per-generation {@code Arena}); this class
 * never closes it. Every read goes through the segment with an explicit offset, so a corrupt offset
 * or an out-of-range node id surfaces as an {@link IndexOutOfBoundsException} from the
 * bounds-checked segment access rather than an unsafe memory read. The {@code graph.bin} CRC is
 * verified upstream (at generation open) before this view is constructed.
 *
 * <p>Reads use the {@code *_UNALIGNED} layouts because degree words and the offset trailer sit at
 * arbitrary (non-8-aligned) file positions.
 */
final class PagedVamanaTopology implements VamanaTopology {

  private final MemorySegment seg;
  private final int numNodes;
  private final int maxDegree;
  private final int medoid;
  private final long trailerStart; // byte offset where the N×int64 offset index begins

  private PagedVamanaTopology(
      MemorySegment seg, int numNodes, int maxDegree, int medoid, long trailerStart) {
    this.seg = seg;
    this.numNodes = numNodes;
    this.maxDegree = maxDegree;
    this.medoid = medoid;
    this.trailerStart = trailerStart;
  }

  /**
   * Parses and validates the v2 {@code graph.bin} header from {@code graphSegment} (which must span
   * exactly the file) and returns a paged view. Does not walk the body or the full offset table —
   * that would defeat lazy paging; gross corruption is caught by the header checks, the {@code
   * offset[0]} sanity check, and per-read bounds checks.
   *
   * @throws IOException if the segment is too small or the header is not a valid v2 Vamana graph
   */
  static PagedVamanaTopology open(MemorySegment graphSegment) throws IOException {
    long size = graphSegment.byteSize();
    if (size < VamanaGraphCodec.HEADER_SIZE) {
      throw new IOException(
          "graph.bin too small for header: " + size + " < " + VamanaGraphCodec.HEADER_SIZE);
    }
    int magic = graphSegment.get(JAVA_INT_UNALIGNED, 0);
    if (magic != FileFormat.MAGIC_GRAPH_VAMANA) {
      throw new IOException(
          String.format(
              "graph.bin (vamana) magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_GRAPH_VAMANA, magic));
    }
    int version = graphSegment.get(JAVA_INT_UNALIGNED, 4);
    if (version != FileFormat.VERSION_GRAPH_VAMANA) {
      throw new IOException(
          "graph.bin (vamana) version mismatch: expected "
              + FileFormat.VERSION_GRAPH_VAMANA
              + ", got "
              + version);
    }
    int headerLength = graphSegment.get(JAVA_INT_UNALIGNED, 8);
    if (headerLength != VamanaGraphCodec.HEADER_SIZE) {
      throw new IOException(
          "graph.bin (vamana) header length mismatch: expected "
              + VamanaGraphCodec.HEADER_SIZE
              + ", got "
              + headerLength);
    }
    int flags = graphSegment.get(JAVA_INT_UNALIGNED, 12);
    if (flags != 0) {
      throw new IOException("graph.bin (vamana) flags must be 0, got " + flags);
    }
    int numNodes = graphSegment.get(JAVA_INT_UNALIGNED, 16);
    if (numNodes < 0) {
      throw new IOException("graph.bin (vamana) numNodes must be >= 0, got " + numNodes);
    }
    int maxDegree = graphSegment.get(JAVA_INT_UNALIGNED, 20);
    if (maxDegree <= 0) {
      throw new IOException("graph.bin (vamana) maxDegree must be positive, got " + maxDegree);
    }
    int medoid = graphSegment.get(JAVA_INT_UNALIGNED, 24);

    if (numNodes == 0) {
      if (size != VamanaGraphCodec.HEADER_SIZE) {
        throw new IOException(
            "graph.bin (vamana) empty graph must be header-only, got " + size + " bytes");
      }
      return new PagedVamanaTopology(graphSegment, 0, maxDegree, -1, VamanaGraphCodec.HEADER_SIZE);
    }

    if (medoid < 0 || medoid >= numNodes) {
      throw new IOException(
          "graph.bin (vamana) medoid " + medoid + " out of range [0, " + numNodes + ")");
    }
    long trailerBytes = 8L * numNodes;
    if (size < VamanaGraphCodec.HEADER_SIZE + trailerBytes) {
      throw new IOException(
          "graph.bin (vamana) too small for "
              + numNodes
              + "-node offset trailer: "
              + size
              + " bytes");
    }
    long trailerStart = size - trailerBytes;
    long firstOffset = graphSegment.get(JAVA_LONG_UNALIGNED, trailerStart);
    if (firstOffset != VamanaGraphCodec.HEADER_SIZE) {
      throw new IOException(
          "graph.bin (vamana) offset[0] must be "
              + VamanaGraphCodec.HEADER_SIZE
              + ", got "
              + firstOffset);
    }
    return new PagedVamanaTopology(graphSegment, numNodes, maxDegree, medoid, trailerStart);
  }

  @Override
  public int size() {
    return numNodes;
  }

  @Override
  public int medoid() {
    return medoid;
  }

  @Override
  public int maxDegree() {
    return maxDegree;
  }

  @Override
  public int neighbors(int nodeId, int[] out) {
    long off = seg.get(JAVA_LONG_UNALIGNED, trailerStart + 8L * nodeId);
    int degree = seg.get(JAVA_INT_UNALIGNED, off);
    long base = off + 4;
    for (int i = 0; i < degree; i++) {
      out[i] = seg.get(JAVA_INT_UNALIGNED, base + 4L * i);
    }
    return degree;
  }
}
