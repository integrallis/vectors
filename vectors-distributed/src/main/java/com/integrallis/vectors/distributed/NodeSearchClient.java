package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.SearchResult;

/**
 * Thin call abstraction over a single node's search endpoint.
 *
 * <p>In-process implementation ({@code InProcessNodeSearchClient}) is used for tests; a gRPC or
 * HTTP implementation would be used in production.
 *
 * <p>Implementations must be thread-safe: the {@link ScatterGatherExecutor} calls {@link #search}
 * from multiple virtual threads concurrently.
 */
public interface NodeSearchClient {

  /**
   * Executes a local search on the target node.
   *
   * @param request the search request, specifying target node, query, cluster ids, k, and min score
   * @return search results from this node (never null; may have empty hits)
   */
  SearchResult search(LocalSearchRequest request);

  /**
   * Returns the number of documents in the committed generation on this node.
   *
   * <p>Used by {@link DistributedVectorCollection#size()} for cluster-wide size aggregation.
   *
   * @return non-negative count
   */
  int size();

  /**
   * Returns the physical (including tombstoned) document count on this node.
   *
   * @return non-negative count
   */
  int physicalSize();
}
