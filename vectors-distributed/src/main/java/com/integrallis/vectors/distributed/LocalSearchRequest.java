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
