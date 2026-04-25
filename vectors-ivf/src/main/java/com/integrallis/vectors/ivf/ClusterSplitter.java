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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.cluster.KMeans;
import java.util.Optional;

/**
 * Decides when a cluster should be split and performs bisecting K-Means to produce two child {@link
 * ClusterPartition}s.
 *
 * <p>The split criterion is derived from the Quake cost model: a cluster is worth splitting when
 * its scan cost (proportional to size) exceeds the routing overhead cost of adding one more routing
 * level. The threshold {@code minSplitSize} is the crossover point where scan work per query
 * exceeds the routing overhead; the default factory {@link #forRootK(int)} sets this to {@code k}
 * (the root cluster count), because routing through {@code k} centroids costs the same as scanning
 * {@code k} vectors.
 *
 * <p>Split mechanism: bisecting K-Means (k=2) on the cluster's assigned vectors. Global ordinals
 * are preserved — child {@link ClusterPartition} ordinals remain indices into the original {@code
 * float[][] vectors} array.
 */
public final class ClusterSplitter {

  private final int minSplitSize;
  private final int maxIter;
  private final long seed;

  /**
   * Creates a splitter with explicit parameters.
   *
   * @param minSplitSize split threshold; must be ≥ 2
   * @param maxIter maximum K-Means iterations for the bisecting step
   * @param seed RNG seed for K-Means++ initialisation
   */
  public ClusterSplitter(int minSplitSize, int maxIter, long seed) {
    if (minSplitSize < 2)
      throw new IllegalArgumentException("minSplitSize must be >= 2, got " + minSplitSize);
    this.minSplitSize = minSplitSize;
    this.maxIter = maxIter;
    this.seed = seed;
  }

  /**
   * Factory: derives split threshold from the root cluster count {@code k}.
   *
   * <p>Quake cost model: split when {@code size >= k}. Routing through {@code k} centroids costs
   * the same as scanning {@code k} vectors, so any cluster larger than {@code k} yields a net
   * saving when split.
   */
  public static ClusterSplitter forRootK(int k) {
    return new ClusterSplitter(Math.max(k, 2), 30, 42L);
  }

  /**
   * Quake cost of scanning a cluster: proportional to its size (number of distance computations).
   *
   * <p>Note: {@code cost(left) + cost(right) == cost(parent)} holds for any bisect. The benefit of
   * splitting comes from only probing ONE child per query after routing — halving the scan work.
   */
  public long cost(int clusterSize) {
    return clusterSize;
  }

  /**
   * Returns {@code true} when scanning this cluster costs more than the routing overhead of one
   * additional level — i.e., when {@code partition.size() >= minSplitSize}.
   */
  public boolean shouldSplit(ClusterPartition partition) {
    return partition.size() >= minSplitSize;
  }

  /**
   * Splits {@code partition} into exactly two child partitions using bisecting K-Means.
   *
   * <p>Child ordinals are global (indices into {@code allVectors}). Child cluster IDs are derived
   * from the parent: {@code parentId * 2 + 1} (left) and {@code parentId * 2 + 2} (right).
   *
   * @param partition the cluster to split
   * @param allVectors the full vector dataset (indexed by global ordinal)
   * @param metric similarity function used for centroid distance during training
   * @param seedOffset added to the constructor seed so repeated calls yield different
   *     initialisations
   * @return the two child partitions, or empty if the cluster has fewer than 2 vectors
   */
  public Optional<ClusterPartition[]> split(
      ClusterPartition partition,
      float[][] allVectors,
      SimilarityFunction metric,
      long seedOffset) {

    int[] ordinals = partition.ordinals();
    if (ordinals.length < 2) return Optional.empty();

    // Extract sub-vectors (by global ordinal) for bisecting K-Means
    float[][] subVectors = new float[ordinals.length][];
    for (int i = 0; i < ordinals.length; i++) {
      subVectors[i] = allVectors[ordinals[i]];
    }

    float[][] centroids = KMeans.train(subVectors, 2, maxIter, seed + seedOffset);
    int[] subAssignments = KMeans.assign(subVectors, centroids);

    // Count sizes
    int sizeLeft = 0;
    for (int a : subAssignments) if (a == 0) sizeLeft++;
    int sizeRight = ordinals.length - sizeLeft;

    // Partition ordinals into two child arrays
    int[] leftOrdinals = new int[sizeLeft];
    int[] rightOrdinals = new int[sizeRight];
    int li = 0, ri = 0;
    for (int i = 0; i < ordinals.length; i++) {
      if (subAssignments[i] == 0) leftOrdinals[li++] = ordinals[i];
      else rightOrdinals[ri++] = ordinals[i];
    }

    int parentId = partition.clusterId();
    return Optional.of(
        new ClusterPartition[] {
          ClusterPartition.of(parentId * 2 + 1, centroids[0], leftOrdinals),
          ClusterPartition.of(parentId * 2 + 2, centroids[1], rightOrdinals)
        });
  }

  /** The configured minimum cluster size at which splitting becomes beneficial. */
  public int minSplitSize() {
    return minSplitSize;
  }
}
