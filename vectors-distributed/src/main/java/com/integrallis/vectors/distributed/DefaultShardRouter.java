package com.integrallis.vectors.distributed;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Default {@link ShardRouter} implementation backed by a {@link ShardOwnership} map.
 *
 * <p>Groups cluster ids by their primary node and emits one {@link LocalSearchRequest} per node.
 * When {@code clusterIds} is empty the router broadcasts a request (with empty clusterIds array) to
 * every node that owns at least one cluster.
 */
public final class DefaultShardRouter implements ShardRouter {

  private final ShardOwnership ownership;
  private final List<NodeId> allNodes;
  private final float minScore;

  /**
   * @param ownership the shard ownership map
   * @param allNodes all known live nodes (used for broadcast when clusterIds is empty)
   * @param minScore minimum score threshold forwarded to each node
   */
  public DefaultShardRouter(ShardOwnership ownership, List<NodeId> allNodes, float minScore) {
    this.ownership = Objects.requireNonNull(ownership, "ownership must not be null");
    this.allNodes = List.copyOf(Objects.requireNonNull(allNodes, "allNodes must not be null"));
    this.minScore = minScore;
  }

  /** Convenience constructor with no minimum score. */
  public DefaultShardRouter(ShardOwnership ownership, List<NodeId> allNodes) {
    this(ownership, allNodes, -Float.MAX_VALUE);
  }

  @Override
  public List<LocalSearchRequest> plan(float[] query, int[] clusterIds, int k) {
    Objects.requireNonNull(query, "query must not be null");
    Objects.requireNonNull(clusterIds, "clusterIds must not be null");

    if (clusterIds.length == 0) {
      // Broadcast: send to every node with an empty clusterIds (search all)
      List<LocalSearchRequest> requests = new ArrayList<>(allNodes.size());
      for (NodeId node : allNodes) {
        Set<Integer> owned = ownership.clustersFor(node);
        if (!owned.isEmpty()) {
          requests.add(new LocalSearchRequest(node, query, new int[0], k, minScore));
        }
      }
      return requests;
    }

    // Group cluster ids by primary node
    Map<NodeId, List<Integer>> nodeToIds = new HashMap<>();
    for (int clusterId : clusterIds) {
      NodeId primary = ownership.primaryFor(clusterId);
      nodeToIds.computeIfAbsent(primary, n -> new ArrayList<>()).add(clusterId);
    }

    List<LocalSearchRequest> requests = new ArrayList<>(nodeToIds.size());
    for (Map.Entry<NodeId, List<Integer>> entry : nodeToIds.entrySet()) {
      List<Integer> ids = entry.getValue();
      int[] arr = new int[ids.size()];
      for (int i = 0; i < ids.size(); i++) {
        arr[i] = ids.get(i);
      }
      requests.add(new LocalSearchRequest(entry.getKey(), query, arr, k, minScore));
    }
    return requests;
  }
}
