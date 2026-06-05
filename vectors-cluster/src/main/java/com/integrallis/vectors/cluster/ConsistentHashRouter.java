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
package com.integrallis.vectors.cluster;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Maps a document id to one of {@code N} shards using consistent hashing.
 *
 * <p>Each shard is placed at {@code virtualNodesPerShard} positions on a hash ring keyed by the MD5
 * hash of {@code "shard-<index>-<v>"}. A document id is routed to the shard whose first virtual
 * position is {@code >= hash(documentId)} (clockwise wrap-around). Consistent hashing — rather than
 * plain {@code hash(id) % N} — means that growing or shrinking the shard count only remaps {@code
 * O(keys / N)} document ids instead of nearly all of them, which is what gives the sharded write
 * tier a clean path to adding shards later.
 *
 * <p>Resharding is <b>not</b> performed in place: this router is immutable, and changing the shard
 * count produces a new router whose routing differs for the remapped slice. Documents already
 * written to a shard are not migrated automatically; a corpus re-ingest (or an external copy of the
 * affected slice) is required to honour the new mapping.
 *
 * <p>Instances are immutable and thread-safe.
 */
public final class ConsistentHashRouter {

  /** Default number of virtual ring positions per shard — high enough for an even spread. */
  public static final int DEFAULT_VIRTUAL_NODES_PER_SHARD = 200;

  private final int shardCount;
  private final int virtualNodesPerShard;
  private final NavigableMap<Long, Integer> ring = new TreeMap<>();

  /**
   * Builds a router over {@code shardCount} shards with {@link #DEFAULT_VIRTUAL_NODES_PER_SHARD}
   * virtual nodes each.
   *
   * @param shardCount number of shards; must be positive
   */
  public ConsistentHashRouter(int shardCount) {
    this(shardCount, DEFAULT_VIRTUAL_NODES_PER_SHARD);
  }

  /**
   * Builds a router over {@code shardCount} shards.
   *
   * @param shardCount number of shards; must be positive
   * @param virtualNodesPerShard ring positions per shard; must be positive. Larger values give a
   *     more even key distribution at the cost of a larger ring.
   */
  public ConsistentHashRouter(int shardCount, int virtualNodesPerShard) {
    if (shardCount <= 0) {
      throw new IllegalArgumentException("shardCount must be positive: " + shardCount);
    }
    if (virtualNodesPerShard <= 0) {
      throw new IllegalArgumentException(
          "virtualNodesPerShard must be positive: " + virtualNodesPerShard);
    }
    this.shardCount = shardCount;
    this.virtualNodesPerShard = virtualNodesPerShard;
    for (int shard = 0; shard < shardCount; shard++) {
      for (int v = 0; v < virtualNodesPerShard; v++) {
        ring.put(hash("shard-" + shard + "-" + v), shard);
      }
    }
  }

  /**
   * Routes a document id to its owning shard index.
   *
   * @param documentId the document id; must not be null
   * @return the owning shard index in {@code [0, shardCount)}
   */
  public int route(String documentId) {
    if (documentId == null) {
      throw new IllegalArgumentException("documentId must not be null");
    }
    long h = hash(documentId);
    var entry = ring.ceilingEntry(h);
    if (entry == null) {
      entry = ring.firstEntry(); // wrap around the ring
    }
    return entry.getValue();
  }

  /** Returns the number of shards this router distributes over. */
  public int shardCount() {
    return shardCount;
  }

  /** Returns the number of virtual ring positions per shard. */
  public int virtualNodesPerShard() {
    return virtualNodesPerShard;
  }

  private static long hash(String key) {
    try {
      MessageDigest md = MessageDigest.getInstance("MD5");
      byte[] bytes = md.digest(key.getBytes(StandardCharsets.UTF_8));
      // Fold the first 8 bytes into a long (big-endian).
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
