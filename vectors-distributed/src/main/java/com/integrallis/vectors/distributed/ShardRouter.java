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
