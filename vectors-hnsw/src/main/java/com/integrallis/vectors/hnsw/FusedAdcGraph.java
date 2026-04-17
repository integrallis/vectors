package com.integrallis.vectors.hnsw;

import java.util.List;

/**
 * An HNSW graph variant where layer-0 neighbor lists carry interleaved PQ codes for Fused ADC
 * scoring.
 *
 * <p>Upper layers ({@code level >= 1}) use the same {@link NeighborArray} storage as a standard
 * {@link HnswGraph}: they are sparsely populated (M neighbors, not 2M) and visited far less often
 * than layer 0, so the cache-locality benefit of code interleaving is marginal there. Layer 0 —
 * visited for every search — is where Fused ADC pays off.
 *
 * <p>During construction, the builder first builds a standard {@link HnswGraph}, then converts each
 * layer-0 neighbor list into a {@link FusedAdcNeighborList} by encoding each neighbor's vector with
 * the trained PQ quantizer. This two-phase approach lets the existing {@link HnswGraphBuilder}
 * drive the graph construction without modification.
 */
public final class FusedAdcGraph {

  private final FusedAdcNeighborList[] layer0; // layer0[nodeId]
  private final HnswGraph upperGraph; // upper layers reuse the standard HnswGraph
  private final int size;

  /**
   * Constructs a FusedAdcGraph by replacing layer-0 neighbor lists with fused lists.
   *
   * @param upperGraph the fully built {@link HnswGraph} (layer 0 content is replaced)
   * @param layer0Fused per-node fused neighbor lists for layer 0
   */
  public FusedAdcGraph(HnswGraph upperGraph, FusedAdcNeighborList[] layer0Fused) {
    this.upperGraph = upperGraph;
    this.layer0 = layer0Fused;
    this.size = upperGraph.size();
  }

  /** Returns the number of nodes in the graph. */
  public int size() {
    return size;
  }

  /** Returns the entry node ID (global entry point for search). */
  public int entryNode() {
    return upperGraph.entryNode();
  }

  /** Returns the maximum level present in the graph. */
  public int maxLevel() {
    return upperGraph.maxLevel();
  }

  /**
   * Returns the fused neighbor list for the given node at layer 0.
   *
   * @param nodeId the node whose neighbors are requested
   * @return the fused ADC neighbor list for {@code nodeId}
   */
  public FusedAdcNeighborList fusedNeighbors(int nodeId) {
    return layer0[nodeId];
  }

  /**
   * Returns the standard neighbor array for the given node at an upper layer.
   *
   * @param nodeId the node whose neighbors are requested
   * @param level the layer level (must be &ge; 1)
   * @return the neighbor array from the underlying {@link HnswGraph}
   */
  public NeighborArray upperNeighbors(int nodeId, int level) {
    return upperGraph.getNeighbors(nodeId, level);
  }

  /**
   * Returns the underlying {@link HnswGraph} for codec / serialization access.
   *
   * @return the backing upper-layer graph
   */
  public HnswGraph upperGraph() {
    return upperGraph;
  }

  /**
   * Returns the list of all fused layer-0 neighbor lists, one per node.
   *
   * @return unmodifiable view of the layer-0 fused lists (may contain nulls for uninitialized
   *     nodes)
   */
  public List<FusedAdcNeighborList> layer0AsList() {
    return List.of(layer0);
  }
}
