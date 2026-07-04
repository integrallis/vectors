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
 * whose positions are derived from the node id (hashed with MurmurHash3, 64-bit) mixed with the
 * virtual-node index {@code i} via {@code fmix64}. A cluster id is assigned to the node whose first
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

    // Populate the ring with virtual nodes. Hash the node id once, then mix in the virtual-node
    // index via fmix64 instead of building (and re-hashing) a "<nodeId>-<i>" string per position.
    for (NodeId node : nodes) {
      long nodeHash = murmur3(node.id().getBytes(StandardCharsets.UTF_8));
      for (int i = 0; i < VIRTUAL_NODES_PER_NODE; i++) {
        ring.put(fmix64(nodeHash + (i + 1) * 0x9E3779B97F4A7C15L), node);
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
    long h = fmix64(CLUSTER_SALT + clusterId);
    Map.Entry<Long, NodeId> entry = ring.ceilingEntry(h);
    if (entry == null) {
      entry = ring.firstEntry(); // wrap around
    }
    return entry.getValue();
  }

  /**
   * Salt that separates cluster-id ring keys from node ring keys in the shared 64-bit space; the
   * exact value is arbitrary but fixed for determinism.
   */
  private static final long CLUSTER_SALT = 0xC1057E12ABCD1234L;

  /**
   * MurmurHash3 128-bit (x64) over {@code data}, returning the first 64-bit half.
   *
   * <p>Replaces a per-call {@link java.security.MessageDigest} MD5 lookup: cluster ownership needs
   * a well-distributed deterministic mapping, not collision resistance. NOTE: this is not
   * compatible with the previous MD5 mapping; ring assignments differ, so a ring built with this
   * version must not be mixed with MD5-built placement without a re-ingest.
   */
  private static long murmur3(byte[] data) {
    final long c1 = 0x87c37b91114253d5L;
    final long c2 = 0x4cf5ad432745937fL;
    final int length = data.length;
    final int nblocks = length >> 4;
    long h1 = 0L;
    long h2 = 0L;
    for (int i = 0; i < nblocks; i++) {
      final int base = i << 4;
      long k1 = getLongLE(data, base);
      long k2 = getLongLE(data, base + 8);
      k1 *= c1;
      k1 = Long.rotateLeft(k1, 31);
      k1 *= c2;
      h1 ^= k1;
      h1 = Long.rotateLeft(h1, 27);
      h1 += h2;
      h1 = h1 * 5 + 0x52dce729;
      k2 *= c2;
      k2 = Long.rotateLeft(k2, 33);
      k2 *= c1;
      h2 ^= k2;
      h2 = Long.rotateLeft(h2, 31);
      h2 += h1;
      h2 = h2 * 5 + 0x38495ab5;
    }
    long k1 = 0L;
    long k2 = 0L;
    final int tail = nblocks << 4;
    final int rem = length & 15;
    for (int i = rem - 1; i >= 8; i--) {
      k2 ^= (long) (data[tail + i] & 0xff) << (8 * (i - 8));
    }
    if (rem > 8) {
      k2 *= c2;
      k2 = Long.rotateLeft(k2, 33);
      k2 *= c1;
      h2 ^= k2;
    }
    for (int i = Math.min(rem, 8) - 1; i >= 0; i--) {
      k1 ^= (long) (data[tail + i] & 0xff) << (8 * i);
    }
    if (rem > 0) {
      k1 *= c1;
      k1 = Long.rotateLeft(k1, 31);
      k1 *= c2;
      h1 ^= k1;
    }
    h1 ^= length;
    h2 ^= length;
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    return h1;
  }

  private static long getLongLE(byte[] b, int i) {
    return (b[i] & 0xffL)
        | ((b[i + 1] & 0xffL) << 8)
        | ((b[i + 2] & 0xffL) << 16)
        | ((b[i + 3] & 0xffL) << 24)
        | ((b[i + 4] & 0xffL) << 32)
        | ((b[i + 5] & 0xffL) << 40)
        | ((b[i + 6] & 0xffL) << 48)
        | ((b[i + 7] & 0xffL) << 56);
  }

  /** 64-bit avalanche finalizer (MurmurHash3 fmix64); a bijection with excellent bit diffusion. */
  private static long fmix64(long k) {
    k ^= k >>> 33;
    k *= 0xff51afd7ed558ccdL;
    k ^= k >>> 33;
    k *= 0xc4ceb9fe1a85ec53L;
    k ^= k >>> 33;
    return k;
  }
}
