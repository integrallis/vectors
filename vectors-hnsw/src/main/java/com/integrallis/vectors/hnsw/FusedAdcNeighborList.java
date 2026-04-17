package com.integrallis.vectors.hnsw;

/**
 * A neighbor list that stores PQ-encoded vectors <em>adjacent to</em> the neighbor IDs in a single
 * contiguous structure, matching the Fused ADC pattern described in the JVector paper.
 *
 * <p>Layout per entry: {@code [int nodeId, byte[M] codes]}. Storing codes next to IDs means that
 * when the CPU fetches a neighbor-ID cache line it simultaneously fetches the codes needed to score
 * that neighbor — eliminating the random-access hop to the global vector matrix.
 *
 * <p>This class is intentionally simple and immutable; it is built once during index construction
 * and read many times during search. Thread-safe by design (no mutable state after construction).
 */
public final class FusedAdcNeighborList {

  private final int[] nodeIds;
  private final byte[][] codes; // codes[i] is the M-byte PQ code for nodeIds[i]
  private final int size;

  /**
   * Constructs a fused neighbor list.
   *
   * @param nodeIds the neighbor node IDs (in descending-score order from the builder)
   * @param codes the PQ codes for each neighbor; {@code codes[i]} corresponds to {@code nodeIds[i]}
   */
  public FusedAdcNeighborList(int[] nodeIds, byte[][] codes) {
    if (nodeIds.length != codes.length) {
      throw new IllegalArgumentException(
          "nodeIds.length=" + nodeIds.length + " != codes.length=" + codes.length);
    }
    this.nodeIds = nodeIds.clone();
    this.codes = new byte[codes.length][];
    for (int i = 0; i < codes.length; i++) {
      this.codes[i] = codes[i].clone();
    }
    this.size = nodeIds.length;
  }

  /** Returns the number of neighbors in this list. */
  public int size() {
    return size;
  }

  /** Returns the neighbor node ID at position {@code i}. */
  public int nodeId(int i) {
    return nodeIds[i];
  }

  /**
   * Scores neighbor {@code i} using the precomputed ADC lookup table.
   *
   * <p>The lookup is O(M) integer-indexed float additions — one per PQ subspace — with no
   * additional memory indirection beyond the local {@code codes[i]} array that was prefetched when
   * the CPU fetched this neighbor's entry.
   *
   * @param i the neighbor index (0-based)
   * @param table the precomputed ADC table returned by {@link
   *     com.integrallis.vectors.quantization.ProductQuantizer#buildADCTable}
   * @return approximate distance/similarity for this neighbor
   */
  public float adcScore(int i, float[][] table) {
    byte[] code = codes[i];
    float sum = 0f;
    for (int m = 0; m < code.length; m++) {
      sum += table[m][code[m] & 0xFF];
    }
    return sum;
  }
}
