package com.integrallis.vectors.distributed;

import java.util.List;

/**
 * Routes a set of IVF cluster ids (produced by a BuoyIndex routing step) to the nodes that own
 * those clusters, producing one {@link LocalSearchRequest} per involved node.
 *
 * <p>When multiple cluster ids map to the same node, they are merged into a single request so the
 * node executes one call and searches all its assigned clusters in one pass.
 */
public interface ShardRouter {

  /**
   * Builds an execution plan for a query.
   *
   * @param query the raw query vector
   * @param clusterIds the IVF cluster ids to search (from BuoyIndex routing); empty means broadcast
   *     to all clusters
   * @param k number of nearest neighbours to return from each node
   * @return one {@link LocalSearchRequest} per target node, in no guaranteed order
   */
  List<LocalSearchRequest> plan(float[] query, int[] clusterIds, int k);
}
