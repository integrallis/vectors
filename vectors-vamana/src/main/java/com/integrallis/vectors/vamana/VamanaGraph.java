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
package com.integrallis.vectors.vamana;

import java.util.Objects;

/**
 * Single-layer flat graph for the Vamana index. Each node has a neighbor list backed by a {@link
 * NeighborArray} with capacity {@code maxDegree + 1} (the +1 allows temporary overflow during
 * backlink insertion before pruning).
 *
 * <p>The entry point is the <b>medoid</b> — the dataset point closest to the centroid — which
 * provides a good starting point for greedy search.
 */
public final class VamanaGraph {

  private final NeighborArray[] neighbors;
  private final int maxDegree;
  private int medoid;
  private int size;

  /**
   * Creates an empty graph with capacity for the given number of nodes.
   *
   * @param maxNodes maximum number of nodes the graph can hold
   * @param maxDegree maximum number of neighbors per node (R in DiskANN paper)
   */
  public VamanaGraph(int maxNodes, int maxDegree) {
    if (maxNodes <= 0) {
      throw new IllegalArgumentException("maxNodes must be positive: " + maxNodes);
    }
    if (maxDegree <= 0) {
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    }
    this.neighbors = new NeighborArray[maxNodes];
    this.maxDegree = maxDegree;
    this.medoid = 0;
    this.size = 0;
  }

  /**
   * Initializes the neighbor list for a node. Must be called exactly once per node before adding
   * neighbors.
   *
   * @param nodeId the node to initialize
   * @throws IllegalStateException if this node has already been initialized
   */
  public void initNode(int nodeId) {
    Objects.checkIndex(nodeId, neighbors.length);
    if (neighbors[nodeId] != null) {
      throw new IllegalStateException(
          "Node " + nodeId + " is already initialized; initNode() must be called exactly once");
    }
    // +1 capacity for temporary overflow during backlink insertion before pruning
    neighbors[nodeId] = new NeighborArray(maxDegree + 1);
    size = Math.max(size, nodeId + 1);
  }

  /**
   * Returns the neighbor list for the given node.
   *
   * @param nodeId the node to query
   * @return the neighbor array (never null)
   * @throws IllegalStateException if this node has not been initialized via {@link #initNode(int)}
   */
  public NeighborArray getNeighbors(int nodeId) {
    Objects.checkIndex(nodeId, size);
    NeighborArray na = neighbors[nodeId];
    if (na == null) {
      throw new IllegalStateException(
          "Node " + nodeId + " has not been initialized; call initNode(" + nodeId + ") first");
    }
    return na;
  }

  /** Returns the medoid (entry point for search). */
  public int medoid() {
    return medoid;
  }

  /** Sets the medoid (entry point for search). */
  public void setMedoid(int nodeId) {
    Objects.checkIndex(nodeId, size);
    this.medoid = nodeId;
  }

  /** Returns the maximum degree (R) — the maximum number of neighbors per node. */
  public int maxDegree() {
    return maxDegree;
  }

  /** Returns the number of initialized nodes. */
  public int size() {
    return size;
  }
}
