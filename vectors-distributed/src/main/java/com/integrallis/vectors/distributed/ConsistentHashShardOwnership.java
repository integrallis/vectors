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

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

/**
 * Consistent-hash implementation of {@link ShardOwnership}.
 *
 * <p>Each real node is placed at {@value #VIRTUAL_NODES_PER_NODE} virtual positions on a hash ring
 * keyed by MD5 hash of {@code "<nodeId>-<i>"}. A cluster id is assigned to the node whose first
 * virtual position is ≥ {@code hash(clusterId)} (clockwise wrap-around). Phase 3: RF=1, so {@link
 * #replicasFor} always returns an empty list.
 */
public final class ConsistentHashShardOwnership implements ShardOwnership {

  /**
   * Virtual nodes per real node on the consistent-hash ring.
   *
   * <p>Why 100: the standard deviation of the per-node ownership share scales as {@code
   * 1/sqrt(V*N)} where V is virtual-nodes-per-real-node and N is real nodes. At V=100, N=8 (a
   * typical cluster) the load imbalance across the ring is roughly ±4% — well below the threshold
   * at which a single hot shard would dominate end-to-end query latency. Bumping to 200 trims that
   * further at the cost of a linear growth in the routing TreeMap memory + log-N lookup constant;
   * 100 is the long-standing default used by Cassandra and Riak and is a safe starting point.
   * {@code vectors-cluster.ConsistentHashRouter} uses 200 by default because its shards are
   * typically fewer and the linear cost is negligible.
   */
  private static final int VIRTUAL_NODES_PER_NODE = 100;

  private final int totalClusters;
  private final NavigableMap<Long, NodeId> ring = new TreeMap<>();
  private final Map<NodeId, Set<Integer>> nodeToClusterIds;

  /**
   * Builds the ownership map by distributing {@code totalClusters} IVF cluster ids across the
   * provided nodes using consistent hashing.
   *
   * @param nodes the live nodes (order is not significant)
   * @param totalClusters total number of IVF clusters to distribute
   */
  public ConsistentHashShardOwnership(Collection<NodeId> nodes, int totalClusters) {
    if (nodes == null || nodes.isEmpty()) {
      throw new IllegalArgumentException("nodes must not be null or empty");
    }
    if (totalClusters <= 0) {
      throw new IllegalArgumentException("totalClusters must be positive: " + totalClusters);
    }
    this.totalClusters = totalClusters;

    // Populate the ring with virtual nodes
    for (NodeId node : nodes) {
      for (int i = 0; i < VIRTUAL_NODES_PER_NODE; i++) {
        long hash = hash(node.id() + "-" + i);
        ring.put(hash, node);
      }
    }

    // Assign each cluster id to its primary node
    Map<NodeId, Set<Integer>> tmp = new HashMap<>();
    for (NodeId node : nodes) {
      tmp.put(node, new HashSet<>());
    }
    for (int clusterId = 0; clusterId < totalClusters; clusterId++) {
      NodeId primary = resolveNode(clusterId);
      tmp.get(primary).add(clusterId);
    }
    // Make inner sets unmodifiable
    Map<NodeId, Set<Integer>> immutable = new HashMap<>();
    tmp.forEach((node, set) -> immutable.put(node, Collections.unmodifiableSet(set)));
    this.nodeToClusterIds = Collections.unmodifiableMap(immutable);
  }

  @Override
  public NodeId primaryFor(int clusterId) {
    if (clusterId < 0 || clusterId >= totalClusters) {
      throw new IllegalArgumentException(
          "clusterId out of range [0, " + totalClusters + "): " + clusterId);
    }
    return resolveNode(clusterId);
  }

  @Override
  public List<NodeId> replicasFor(int clusterId) {
    // Phase 3: RF=1, no replicas
    return List.of();
  }

  @Override
  public Set<Integer> clustersFor(NodeId node) {
    Set<Integer> owned = nodeToClusterIds.get(node);
    return owned != null ? owned : Set.of();
  }

  @Override
  public int totalClusters() {
    return totalClusters;
  }

  // --- private helpers ---

  private NodeId resolveNode(int clusterId) {
    long h = hash("cluster-" + clusterId);
    Map.Entry<Long, NodeId> entry = ring.ceilingEntry(h);
    if (entry == null) {
      entry = ring.firstEntry(); // wrap around
    }
    return entry.getValue();
  }

  private static long hash(String key) {
    try {
      MessageDigest md = MessageDigest.getInstance("MD5");
      byte[] bytes = md.digest(key.getBytes(StandardCharsets.UTF_8));
      // Use first 8 bytes as a long (unsigned interpretation via shift)
      long h = 0;
      for (int i = 0; i < 8; i++) {
        h = (h << 8) | (bytes[i] & 0xFFL);
      }
      return h;
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("MD5 not available", e);
    }
  }
}
