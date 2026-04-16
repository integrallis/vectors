package com.integrallis.vectors.distributed;

import java.util.Set;

/**
 * Registry that resolves a {@link NodeId} to its {@link NodeSearchClient}.
 *
 * <p>In-process implementation ({@code InProcessNodeDirectory}) is used for tests. A production
 * implementation would maintain gRPC channels or HTTP clients per node.
 *
 * <p>Implementations must be thread-safe.
 */
public interface NodeDirectory {

  /**
   * Returns the search client for the given node.
   *
   * @param nodeId the target node
   * @return the client (never null)
   * @throws IllegalArgumentException if the node is not registered
   */
  NodeSearchClient clientFor(NodeId nodeId);

  /**
   * Returns the set of all known node ids registered in this directory.
   *
   * @return unmodifiable set (may be empty)
   */
  Set<NodeId> allNodes();
}
