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

import java.util.Arrays;
import java.util.Objects;

/**
 * A search request destined for a single cluster node.
 *
 * <p>{@code clusterIds} is the subset of IVF cluster identifiers that the target node owns and
 * should search. An empty array signals a broadcast search (search all clusters on the node).
 *
 * @param targetNode the node that should execute this request
 * @param query the raw query vector (stored by reference — caller must not mutate after passing)
 * @param clusterIds cluster ids this node should restrict its search to (empty = search all)
 * @param k number of nearest neighbours to return
 * @param minScore minimum similarity score threshold (use {@code -Float.MAX_VALUE} for none)
 */
public record LocalSearchRequest(
    NodeId targetNode, float[] query, int[] clusterIds, int k, float minScore) {

  public LocalSearchRequest {
    Objects.requireNonNull(targetNode, "targetNode must not be null");
    Objects.requireNonNull(query, "query must not be null");
    Objects.requireNonNull(clusterIds, "clusterIds must not be null");
    if (query.length == 0) {
      throw new IllegalArgumentException("query must not be empty");
    }
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof LocalSearchRequest r)) return false;
    return k == r.k
        && Float.compare(minScore, r.minScore) == 0
        && Objects.equals(targetNode, r.targetNode)
        && Arrays.equals(query, r.query)
        && Arrays.equals(clusterIds, r.clusterIds);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(targetNode, k, minScore);
    result = 31 * result + Arrays.hashCode(query);
    result = 31 * result + Arrays.hashCode(clusterIds);
    return result;
  }

  @Override
  public String toString() {
    return "LocalSearchRequest[targetNode="
        + targetNode
        + ", k="
        + k
        + ", clusterIds="
        + Arrays.toString(clusterIds)
        + ", minScore="
        + minScore
        + "]";
  }
}
