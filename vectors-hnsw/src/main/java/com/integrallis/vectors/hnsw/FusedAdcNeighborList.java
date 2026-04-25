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
package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.VectorUtil;

/**
 * A neighbor list that stores PQ-encoded codes in a single contiguous byte array laid out
 * neighbor-major: the {@code M} codes for the {@code i}-th neighbor occupy {@code packedCodes[i*M
 * .. (i+1)*M)}.
 *
 * <p>This is the layout JVector uses for its Fused ADC graph feature: fetching one cache line from
 * {@code packedCodes} at the origin node's base address typically delivers the codes for multiple
 * neighbors at once, letting a batched ADC scorer read them with stride-1 access and no per-row
 * pointer indirection.
 *
 * <p>The list is immutable — built once during index construction and read many times during
 * search. Thread-safe by design.
 */
public final class FusedAdcNeighborList {

  private final int[] nodeIds;
  private final byte[] packedCodes; // neighbor-major: [M bytes for nb0][M bytes for nb1]...
  private final int subspaceCount; // M
  private final int size;

  /**
   * Constructs a fused neighbor list.
   *
   * @param nodeIds the neighbor node IDs (in descending-score order from the builder)
   * @param packedCodes neighbor-major packed PQ codes of length {@code nodeIds.length *
   *     subspaceCount}
   * @param subspaceCount the number of PQ subspaces (M)
   */
  public FusedAdcNeighborList(int[] nodeIds, byte[] packedCodes, int subspaceCount) {
    if (packedCodes.length != nodeIds.length * subspaceCount) {
      throw new IllegalArgumentException(
          "packedCodes.length="
              + packedCodes.length
              + " != nodeIds.length*subspaceCount="
              + (nodeIds.length * subspaceCount));
    }
    this.nodeIds = nodeIds.clone();
    this.packedCodes = packedCodes.clone();
    this.subspaceCount = subspaceCount;
    this.size = nodeIds.length;
  }

  /**
   * Builds a fused neighbor list by copying {@code nodeIds} and pulling each neighbor's M-byte code
   * from {@code allCodes[neighborId]} into the packed layout.
   *
   * @param nodeIds the neighbor IDs in insertion order
   * @param allCodes per-node PQ codes; {@code allCodes[globalId]} holds the M bytes for that node
   * @param subspaceCount M — the number of PQ subspaces
   * @return a fully packed {@link FusedAdcNeighborList}
   */
  public static FusedAdcNeighborList pack(int[] nodeIds, byte[][] allCodes, int subspaceCount) {
    byte[] packed = new byte[nodeIds.length * subspaceCount];
    for (int i = 0; i < nodeIds.length; i++) {
      byte[] code = allCodes[nodeIds[i]];
      System.arraycopy(code, 0, packed, i * subspaceCount, subspaceCount);
    }
    return new FusedAdcNeighborList(nodeIds, packed, subspaceCount);
  }

  /** Returns the number of neighbors in this list. */
  public int size() {
    return size;
  }

  /** Returns the number of PQ subspaces (M) per code. */
  public int subspaceCount() {
    return subspaceCount;
  }

  /** Returns the neighbor node ID at position {@code i}. */
  public int nodeId(int i) {
    return nodeIds[i];
  }

  /**
   * Returns the underlying packed byte buffer of length {@code size() * subspaceCount()} for direct
   * SIMD ADC scoring via {@link VectorUtil#batchAssembleAndSum}. The returned array is the internal
   * storage — callers must not mutate it.
   */
  public byte[] packedCodes() {
    return packedCodes;
  }

  /**
   * Scores neighbor {@code i} using the precomputed ADC lookup {@code table} of shape {@code
   * [M][K]}. Each lookup is O(M) additions with a single sequential scan through {@code
   * packedCodes[i*M..]}.
   *
   * @return the raw partial-sum score (caller maps to a similarity via the appropriate transform)
   */
  public float adcScore(int i, float[][] table) {
    return VectorUtil.assembleAndSum(table, packedCodes, i * subspaceCount, subspaceCount);
  }

  /**
   * Scores all neighbors into {@code out[0..size())} via {@link VectorUtil#batchAssembleAndSum}.
   * The batched kernel is 4-row unrolled — significantly faster than calling {@link #adcScore} in a
   * loop for typical degrees ({@code size >= 4}).
   */
  public void batchAdcScore(float[][] table, float[] out) {
    VectorUtil.batchAssembleAndSum(table, packedCodes, 0, out, size, subspaceCount);
  }
}
