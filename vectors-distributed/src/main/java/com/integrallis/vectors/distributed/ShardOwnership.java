package com.integrallis.vectors.distributed;

import java.util.List;
import java.util.Set;

/**
 * Ownership map: for each IVF cluster id, which node is the primary and which are replicas.
 *
 * <p>Implementations must be thread-safe and immutable for the lifetime of a query.
 */
public interface ShardOwnership {

  /**
   * Returns the primary node responsible for the given cluster id.
   *
   * @param clusterId the IVF cluster ordinal
   * @return primary node (never null)
   * @throws IllegalArgumentException if clusterId is out of range
   */
  NodeId primaryFor(int clusterId);

  /**
   * Returns the replica nodes for the given cluster id. In the Phase 3 single-replica model this
   * always returns an empty list.
   *
   * @param clusterId the IVF cluster ordinal
   * @return unmodifiable list of replica nodes (may be empty)
   */
  List<NodeId> replicasFor(int clusterId);

  /**
   * Returns the set of cluster ids owned (as primary) by the given node.
   *
   * @param node the node to query
   * @return unmodifiable set of cluster ids (may be empty if the node owns none)
   */
  Set<Integer> clustersFor(NodeId node);

  /** Total number of IVF clusters managed by this ownership map. */
  int totalClusters();
}
