/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.ivf;

import java.util.HashMap;
import java.util.Map;

/**
 * Frequency-based tier promotion policy for the T0–T3 storage cascade.
 *
 * <p>The four storage tiers are:
 *
 * <ul>
 *   <li><b>T0</b> — 1-bit RaBitQ (heap); globally replicated, always present. Used for coarse
 *       cluster routing; never managed by this policy.
 *   <li><b>T1</b> — SQ8 (off-heap); hot clusters. Promoted when {@code accessCount ≥ t1Threshold}.
 *       Enables fast approximate rescore without loading full float32 data.
 *   <li><b>T2</b> — Float32 (mmap/SSD); warm clusters. Promoted when {@code accessCount ≥
 *       t2Threshold && accessCount < t1Threshold}. Highest local precision at the cost of disk I/O.
 *   <li><b>T3</b> — Float32 (S3/cloud); cold clusters. Default tier for clusters that have not yet
 *       been promoted.
 * </ul>
 *
 * <p>Promotion is monotone: higher access count never yields a lower tier. Demotion (eviction) is a
 * separate operator concern and is not managed by this class.
 *
 * <p>Usage example:
 *
 * <pre>{@code
 * TierPolicy policy = TierPolicy.defaults();
 * Tier desired = policy.desiredTier(cluster.accessCount());
 * }</pre>
 */
public final class TierPolicy {

  /** The four storage tiers, ordered from fastest/smallest to slowest/largest. */
  public enum Tier {
    /** 1-bit RaBitQ, heap — globally replicated, always available. */
    T0,
    /** SQ8, off-heap — hot clusters promoted for fast rescore. */
    T1,
    /** Float32, mmap/SSD — warm clusters. */
    T2,
    /** Float32, S3/cloud — cold default. */
    T3
  }

  private final int t1Threshold;
  private final int t2Threshold;

  /**
   * Creates a {@code TierPolicy} with explicit thresholds.
   *
   * @param t1Threshold access count at which a cluster is promoted to T1 (SQ8); must be {@code >
   *     t2Threshold}
   * @param t2Threshold access count at which a cluster is promoted to T2 (float32 mmap); must be
   *     {@code ≥ 1}
   */
  public TierPolicy(int t1Threshold, int t2Threshold) {
    if (t2Threshold < 1)
      throw new IllegalArgumentException("t2Threshold must be >= 1, got " + t2Threshold);
    if (t1Threshold <= t2Threshold)
      throw new IllegalArgumentException(
          "t1Threshold must be > t2Threshold, got t1=" + t1Threshold + " t2=" + t2Threshold);
    this.t1Threshold = t1Threshold;
    this.t2Threshold = t2Threshold;
  }

  /**
   * Default policy: promote to T2 after 3 accesses, promote to T1 after 10 accesses.
   *
   * <p>These defaults are calibrated for workloads with ~1 000 clusters where 5% of clusters
   * receive 80% of queries (Zipf distribution).
   */
  public static TierPolicy defaults() {
    return new TierPolicy(10, 3);
  }

  /**
   * Returns the desired {@link Tier} for a cluster with the given access count.
   *
   * <ul>
   *   <li>{@code count < t2Threshold} → {@link Tier#T3}
   *   <li>{@code t2Threshold ≤ count < t1Threshold} → {@link Tier#T2}
   *   <li>{@code count ≥ t1Threshold} → {@link Tier#T1}
   * </ul>
   *
   * @param accessCount total number of times the cluster has been probed since last reset
   * @return the target materialization tier
   */
  public Tier desiredTier(int accessCount) {
    if (accessCount >= t1Threshold) return Tier.T1;
    if (accessCount >= t2Threshold) return Tier.T2;
    return Tier.T3;
  }

  /**
   * Applies the policy to a map of cluster-id → access-count, returning the desired tier for each
   * cluster.
   *
   * @param accessCounts mapping from cluster id to its total access count
   * @return mapping from cluster id to desired {@link Tier}
   */
  public Map<Integer, Tier> applyAll(Map<Integer, Integer> accessCounts) {
    Map<Integer, Tier> result = new HashMap<>(accessCounts.size() * 2);
    for (Map.Entry<Integer, Integer> entry : accessCounts.entrySet()) {
      result.put(entry.getKey(), desiredTier(entry.getValue()));
    }
    return result;
  }

  /** Returns the T1 promotion threshold. */
  public int t1Threshold() {
    return t1Threshold;
  }

  /** Returns the T2 promotion threshold. */
  public int t2Threshold() {
    return t2Threshold;
  }
}
