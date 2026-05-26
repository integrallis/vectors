/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
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
   * Executes a local search with node-call metadata.
   *
   * <p>Transport-backed implementations should use {@code context.bearerToken()} as the
   * node-to-node bearer credential. Implementations that do not require auth may ignore it.
   */
  default SearchResult search(LocalSearchRequest request, NodeCallContext context) {
    return search(request);
  }

  /**
   * Returns the number of documents in the committed generation on this node.
   *
   * <p>Used by {@link DistributedVectorCollection#size()} for cluster-wide size aggregation.
   *
   * @return non-negative count
   */
  int size();

  /** Returns the logical document count with node-call metadata. */
  default int size(NodeCallContext context) {
    return size();
  }

  /**
   * Returns the physical (including tombstoned) document count on this node.
   *
   * @return non-negative count
   */
  int physicalSize();

  /** Returns the physical document count with node-call metadata. */
  default int physicalSize(NodeCallContext context) {
    return physicalSize();
  }
}
