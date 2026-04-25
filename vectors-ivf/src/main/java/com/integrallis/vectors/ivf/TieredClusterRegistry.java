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

import java.util.Collection;
import java.util.Collections;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe registry mapping cluster ids (via {@link HyperDoor#clusterOrdinal()}) to their
 * corresponding {@link TieredCluster} instances.
 *
 * <p>Used by {@link DistributedVectorCollection} to support cross-shard cluster lookups in future
 * distributed execution paths where a {@link HyperDoor} address may arrive from a remote node.
 *
 * <p>All operations are O(1) average case (hash map).
 */
public final class TieredClusterRegistry {

  private final ConcurrentHashMap<Integer, TieredCluster> byClusterId = new ConcurrentHashMap<>();

  /**
   * Registers a {@link TieredCluster} under its {@link HyperDoor#clusterOrdinal()} key. Replaces
   * any previously registered cluster with the same id (e.g., after a commit cycle rebuilds
   * clusters with updated ordinals).
   *
   * @param cluster the cluster to register; must not be {@code null}
   */
  public void register(TieredCluster cluster) {
    byClusterId.put(cluster.hyperDoor().clusterOrdinal(), cluster);
  }

  /**
   * Looks up the {@link TieredCluster} whose cluster id matches {@link HyperDoor#clusterOrdinal()}.
   *
   * @param door the address record identifying the target cluster
   * @return the cluster, or {@link Optional#empty()} if no cluster with that id is registered
   */
  public Optional<TieredCluster> lookup(HyperDoor door) {
    return Optional.ofNullable(byClusterId.get(door.clusterOrdinal()));
  }

  /**
   * Returns the number of clusters currently registered.
   *
   * @return registry size
   */
  public int size() {
    return byClusterId.size();
  }

  /**
   * Returns an unmodifiable view of all registered clusters.
   *
   * @return collection of registered {@link TieredCluster}s
   */
  public Collection<TieredCluster> clusters() {
    return Collections.unmodifiableCollection(byClusterId.values());
  }
}
