/**
 * Sharded write tier (ROADMAP P3.2): a {@link com.integrallis.vectors.db.VectorCollection}-shaped
 * facade that hash-partitions documents by id across {@code N} fully independent {@code vectors-db}
 * shards.
 *
 * <p>{@link com.integrallis.vectors.cluster.ConsistentHashRouter} maps each document id to its
 * owning shard; {@link com.integrallis.vectors.cluster.ClusterVectorCollection} routes writes to
 * the owning shard and scatter-gathers reads across all shards, merging the per-shard top-k
 * results. There is no consensus and no cross-shard atomic commit — shards are independent, which
 * is what gives the tier horizontal write throughput.
 */
package com.integrallis.vectors.cluster;
