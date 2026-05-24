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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Passive-gossip implementation of {@link ClusterMembership}.
 *
 * <p>Models the cluster-wide propagation of membership events and BuoyIndex version hashes without
 * requiring a leader or consensus protocol. This is the in-process simulation of what in a real
 * distributed deployment would be UDP multicast or a gossip protocol (e.g. SWIM/SwimRing).
 *
 * <h2>BuoyIndex gossip protocol</h2>
 *
 * <p>Each node holds a local copy of the BuoyIndex. When a node trains or receives a new index, it
 * calls {@link #announceVersion(String)} with the new version hash. {@code GossipClusterMembership}
 * records the new hash, compares it with the previously known cluster version, and — if different —
 * fans out a {@link MembershipEvent.BuoyIndexUpdated} event to all registered listeners on all
 * nodes. Listeners should reload the BuoyIndex from the shared store (S3 or coordinator) when they
 * receive this event.
 *
 * <h2>Node join / leave</h2>
 *
 * <p>Call {@link #join(NodeId)} and {@link #leave(NodeId)} to simulate nodes entering and leaving
 * the cluster. These fire {@link MembershipEvent.NodeJoined} / {@link MembershipEvent.NodeLeft}
 * events to all listeners.
 *
 * <h2>Thread safety</h2>
 *
 * <p>All methods are thread-safe. Listeners are called on the calling thread (i.e. {@link
 * #announceVersion}, {@link #join}, {@link #leave}); they must be non-blocking.
 */
public final class GossipClusterMembership implements ClusterMembership {

  private static final Logger LOG = LoggerFactory.getLogger(GossipClusterMembership.class);

  /** The set of live nodes — held in a thread-safe, effectively immutable wrapper. */
  private volatile Set<NodeId> liveNodes;

  /** The cluster-wide "current" BuoyIndex version hash (empty string = no index yet). */
  private final AtomicReference<String> clusterVersionHash = new AtomicReference<>("");

  /** All registered change listeners across all nodes sharing this membership instance. */
  private final List<Consumer<MembershipEvent>> listeners = new CopyOnWriteArrayList<>();

  /**
   * Creates a membership view for the given initial set of live nodes.
   *
   * @param initialNodes the nodes that are already live at creation time (must not be null/empty)
   */
  public GossipClusterMembership(Collection<NodeId> initialNodes) {
    Objects.requireNonNull(initialNodes, "initialNodes must not be null");
    if (initialNodes.isEmpty()) {
      throw new IllegalArgumentException("initialNodes must not be empty");
    }
    this.liveNodes = Set.copyOf(initialNodes);
  }

  // ---- ClusterMembership interface ----

  @Override
  public Set<NodeId> liveNodes() {
    return liveNodes;
  }

  @Override
  public void registerChangeListener(Consumer<MembershipEvent> listener) {
    listeners.add(Objects.requireNonNull(listener, "listener must not be null"));
  }

  // ---- Gossip primitives ----

  /**
   * Announces that this node has a new BuoyIndex with the given version hash.
   *
   * <p>If the announced version differs from the cluster-wide known version, a {@link
   * MembershipEvent.BuoyIndexUpdated} event is fanned out to all registered listeners. Idempotent:
   * announcing the same version twice produces only one event.
   *
   * @param newVersionHash opaque version identifier (e.g. SHA-256 of serialised centroid bytes)
   */
  public void announceVersion(String newVersionHash) {
    Objects.requireNonNull(newVersionHash, "newVersionHash must not be null");
    String prev = clusterVersionHash.getAndUpdate(cur -> newVersionHash);
    if (!newVersionHash.equals(prev)) {
      LOG.debug(
          "BuoyIndex version updated: {} → {}", prev.isEmpty() ? "<none>" : prev, newVersionHash);
      fireEvent(new MembershipEvent.BuoyIndexUpdated(newVersionHash));
    }
  }

  /**
   * Returns the current cluster-wide BuoyIndex version hash, or an empty string if no version has
   * been announced yet.
   */
  public String currentVersionHash() {
    return clusterVersionHash.get();
  }

  /**
   * Simulates a node joining the cluster.
   *
   * <p>Adds {@code nodeId} to the live-node set and fires a {@link MembershipEvent.NodeJoined}
   * event. No-op if the node is already live.
   *
   * @param nodeId the joining node
   */
  public void join(NodeId nodeId) {
    Objects.requireNonNull(nodeId, "nodeId must not be null");
    synchronized (this) {
      if (liveNodes.contains(nodeId)) return;
      Set<NodeId> updated = new java.util.HashSet<>(liveNodes);
      updated.add(nodeId);
      liveNodes = Set.copyOf(updated);
    }
    LOG.debug("Node joined: {}", nodeId);
    fireEvent(new MembershipEvent.NodeJoined(nodeId));
  }

  /**
   * Simulates a node leaving or becoming unreachable.
   *
   * <p>Removes {@code nodeId} from the live-node set and fires a {@link MembershipEvent.NodeLeft}
   * event. No-op if the node was not live.
   *
   * @param nodeId the departing node
   */
  public void leave(NodeId nodeId) {
    Objects.requireNonNull(nodeId, "nodeId must not be null");
    synchronized (this) {
      if (!liveNodes.contains(nodeId)) return;
      Set<NodeId> updated = new java.util.HashSet<>(liveNodes);
      updated.remove(nodeId);
      liveNodes = Set.copyOf(updated);
    }
    LOG.debug("Node left: {}", nodeId);
    fireEvent(new MembershipEvent.NodeLeft(nodeId));
  }

  // ---- private ----

  private void fireEvent(MembershipEvent event) {
    List<RuntimeException> errors = new ArrayList<>();
    for (Consumer<MembershipEvent> listener : listeners) {
      try {
        listener.accept(event);
      } catch (RuntimeException e) {
        errors.add(e);
      }
    }
    if (!errors.isEmpty()) {
      RuntimeException first = errors.get(0);
      errors.subList(1, errors.size()).forEach(first::addSuppressed);
      throw first;
    }
  }
}
