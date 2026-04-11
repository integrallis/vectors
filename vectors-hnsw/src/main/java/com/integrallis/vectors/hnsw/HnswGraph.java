package com.integrallis.vectors.hnsw;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-layer graph structure for HNSW indexing. Each node has a neighbor list at layer 0 and
 * optionally at higher layers (assigned by exponential distribution during insertion).
 *
 * <p>Neighbor storage is on-heap using {@link NeighborArray} per node per layer. Layer 0 allows
 * {@code 2*M} connections; upper layers allow {@code M} connections.
 */
public final class HnswGraph {

  private final int maxConnections; // M
  private final int maxConnections0; // 2*M
  private final NeighborArray[] layer0; // layer0[nodeId]
  private final List<NeighborArray[]> upperLayers; // upperLayers.get(l-1)[nodeId]
  private final int[] nodeLevels; // highest layer per node
  private int entryNode;
  private int maxLevel;
  private int size;

  /**
   * Creates an empty graph.
   *
   * @param maxNodes maximum number of nodes the graph can hold
   * @param maxConnections the M parameter (connections per layer for upper layers)
   */
  public HnswGraph(int maxNodes, int maxConnections) {
    if (maxNodes <= 0) {
      throw new IllegalArgumentException("maxNodes must be positive: " + maxNodes);
    }
    if (maxConnections <= 0) {
      throw new IllegalArgumentException("maxConnections must be positive: " + maxConnections);
    }
    this.maxConnections = maxConnections;
    this.maxConnections0 = 2 * maxConnections;
    this.layer0 = new NeighborArray[maxNodes];
    this.upperLayers = new ArrayList<>();
    this.nodeLevels = new int[maxNodes];
    this.entryNode = -1;
    this.maxLevel = -1;
    this.size = 0;
  }

  /** Returns M (max connections for upper layers). */
  public int maxConnections() {
    return maxConnections;
  }

  /** Returns 2*M (max connections for layer 0). */
  public int maxConnections0() {
    return maxConnections0;
  }

  /** Returns the current entry node, or -1 if the graph is empty. */
  public int entryNode() {
    return entryNode;
  }

  /** Returns the highest layer in the graph, or -1 if the graph is empty. */
  public int maxLevel() {
    return maxLevel;
  }

  /** Returns the number of nodes in the graph. */
  public int size() {
    return size;
  }

  /** Returns the highest layer assigned to the given node. */
  public int nodeLevel(int nodeId) {
    return nodeLevels[nodeId];
  }

  /**
   * Returns the neighbor list for the given node at the given layer, or null if the node is not
   * present at that layer.
   */
  public NeighborArray getNeighbors(int nodeId, int layer) {
    if (layer == 0) {
      return layer0[nodeId];
    }
    int upperIndex = layer - 1;
    if (upperIndex >= upperLayers.size()) {
      return null;
    }
    NeighborArray[] layerArray = upperLayers.get(upperIndex);
    if (nodeId >= layerArray.length) {
      return null;
    }
    return layerArray[nodeId];
  }

  /**
   * Initializes a node at the given level, allocating neighbor arrays for layer 0 through the given
   * level. Expands upper layer storage as needed.
   *
   * <p>This method is public so that persistent-snapshot deserialization paths (e.g. {@code
   * com.integrallis.vectors.db.storage.HnswGraphCodec}) can reconstruct a graph from bytes. The
   * normal build path — {@link HnswGraphBuilder} — calls this during Algorithm 1 insertion.
   * External callers that bypass the builder are responsible for populating the resulting neighbor
   * arrays in descending score order and calling {@link #setEntryNode(int, int)} exactly once.
   */
  public void initNode(int nodeId, int level) {
    nodeLevels[nodeId] = level;
    // +1 capacity for temporary overflow during backlink insertion before diversity pruning
    layer0[nodeId] = new NeighborArray(maxConnections0 + 1);

    // Ensure upperLayers has enough layers
    while (upperLayers.size() < level) {
      upperLayers.add(new NeighborArray[layer0.length]);
    }

    // Allocate upper-layer neighbor arrays (+1 for temporary overflow)
    for (int l = 1; l <= level; l++) {
      upperLayers.get(l - 1)[nodeId] = new NeighborArray(maxConnections + 1);
    }
    size++;
  }

  /**
   * Updates the entry node. {@code maxLevel} is only raised, never lowered: the entry node is
   * always the node with the highest level seen so far, and {@code maxLevel} represents the global
   * peak of the graph.
   *
   * <p>This method is public so that persistent-snapshot deserialization paths (e.g. {@code
   * com.integrallis.vectors.db.storage.HnswGraphCodec}) can reconstruct a graph from bytes.
   */
  public void setEntryNode(int nodeId, int level) {
    this.entryNode = nodeId;
    this.maxLevel = Math.max(this.maxLevel, level);
  }
}
