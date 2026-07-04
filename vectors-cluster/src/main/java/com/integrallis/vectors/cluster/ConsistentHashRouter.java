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
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Maps a document id to one of {@code N} shards using consistent hashing.
 *
 * <p>Each shard is placed at {@code virtualNodesPerShard} positions on a hash ring whose positions
 * are derived from the {@code (shard, v)} pair via a 64-bit finalizer (fmix64). A document id is
 * routed to the shard whose first virtual position is {@code >= hash(documentId)} (clockwise
 * wrap-around), where {@code hash} is a fast non-cryptographic hash (MurmurHash3, 64-bit).
 * Consistent hashing — rather than plain {@code hash(id) % N} — means that growing or shrinking the
 * shard count only remaps {@code O(keys / N)} document ids instead of nearly all of them, which is
 * what gives the sharded write tier a clean path to adding shards later.
 *
 * <p>Resharding is <b>not</b> performed in place: this router is immutable, and changing the shard
 * count produces a new router whose routing differs for the remapped slice. Documents already
 * written to a shard are not migrated automatically; a corpus re-ingest (or an external copy of the
 * affected slice) is required to honour the new mapping.
 *
 * <p>Instances are immutable and thread-safe.
 */
public final class ConsistentHashRouter {

  /**
   * Default number of virtual ring positions per shard.
   *
   * <p>Why 200: load imbalance across the ring scales as ~{@code 1/sqrt(V*S)} where V is this value
   * and S is the shard count. A typical write tier has 4-16 shards, so V=200 yields roughly ±2.5%
   * imbalance at S=8 — well inside the noise floor of per-shard ingest latency. {@code
   * vectors-distributed.ConsistentHashShardOwnership} uses V=100 because it operates on physical
   * node counts that are typically larger (more nodes → smaller V suffices for the same imbalance
   * bound).
   */
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
        ring.put(ringPosition(shard, v), shard);
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

  /**
   * Hashes a document id to a 64-bit ring position.
   *
   * <p>Uses MurmurHash3 (128-bit x64 variant, first 64-bit half) rather than a cryptographic
   * digest: routing needs a well-distributed, deterministic mapping, not collision resistance. This
   * avoids a {@link java.security.MessageDigest} lookup + allocation on every {@code route} call,
   * which is on the hot path of every sharded write and read. NOTE: this hash is not compatible
   * with the previous MD5-based mapping — a ring built by this version will place ids differently,
   * so it must not be mixed with data placed by an MD5 build without a re-ingest.
   */
  private static long hash(String key) {
    return murmur3(key.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Deterministic ring position for a shard's {@code v}-th virtual node.
   *
   * <p>Derived directly from the {@code (shard, v)} pair via {@code fmix64} instead of hashing the
   * throwaway string {@code "shard-<shard>-<v>"}. {@code fmix64} is a bijection, so distinct pairs
   * map to distinct, uniformly spread positions; this removes {@code shardCount *
   * virtualNodesPerShard} transient String + UTF-8 byte[] allocations from ring construction. A
   * shard's positions depend only on {@code (shard, v)} and not on {@code shardCount}, preserving
   * the consistent-hashing property that growing the ring only remaps an {@code O(keys /
   * shardCount)} slice.
   */
  private static long ringPosition(int shard, int v) {
    return fmix64(((long) shard << 32) | (v & 0xFFFF_FFFFL));
  }

  /** MurmurHash3 128-bit (x64) over {@code data}, returning the first 64-bit half. */
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
