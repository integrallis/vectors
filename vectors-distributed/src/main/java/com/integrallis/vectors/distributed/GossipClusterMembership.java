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
 * calls {@link #announceVersion(long, String)} with the new index's monotonic generation and
 * content hash. {@code GossipClusterMembership} accepts it only if its generation is newer than the
 * cluster's current one (rejecting stale/reordered announces), and — if accepted — fans out a {@link
 * MembershipEvent.BuoyIndexUpdated} event to all registered listeners on all nodes. Listeners should
 * reload the BuoyIndex from the shared store (S3 or coordinator) when they receive this event.
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

  /**
   * The cluster-wide "current" BuoyIndex version, as a monotonic {@code generation} paired with its
   * opaque content hash. The generation gives a total order so a delayed/stale announce of an older
   * index can be rejected rather than clobbering a newer one (the hash alone is unordered). The
   * sentinel {@code (Long.MIN_VALUE, "")} means "no index yet".
   */
  private final AtomicReference<VersionStamp> clusterVersion =
      new AtomicReference<>(new VersionStamp(Long.MIN_VALUE, ""));

  /** A cluster BuoyIndex version: a monotonic generation plus its opaque content hash. */
  private record VersionStamp(long generation, String hash) {}

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
   * Announces that this node has a new BuoyIndex, identified by a monotonic {@code generation} and
   * its content {@code versionHash}.
   *
   * <p><b>Monotonic guard (max-wins):</b> the announcement is accepted only if its {@code generation}
   * is strictly greater than the cluster's current generation; a stale/delayed announce of an older
   * generation is ignored so it cannot clobber a newer index (which would tell listeners to reload a
   * stale one). On acceptance a {@link MembershipEvent.BuoyIndexUpdated} event is fanned out to all
   * listeners. This is the standard gossip rule (cf. Cassandra generation/version, SWIM incarnation
   * numbers): the caller must assign a strictly increasing generation per committed index (e.g. the
   * collection's monotonic commit generation).
   *
   * <p>Idempotent: re-announcing the current generation produces no event. A different hash at the
   * <em>same</em> generation indicates two distinct indexes claiming one generation — a
   * monotonicity violation upstream — and is logged and ignored (first-writer-wins), never applied.
   *
   * @param generation monotonic, strictly-increasing per committed index (e.g. the commit generation)
   * @param versionHash opaque content identifier (e.g. SHA-256 of serialised centroid bytes)
   */
  public void announceVersion(long generation, String versionHash) {
    Objects.requireNonNull(versionHash, "versionHash must not be null");
    VersionStamp next = new VersionStamp(generation, versionHash);
    VersionStamp prev;
    do {
      prev = clusterVersion.get();
      if (generation < prev.generation()) {
        LOG.debug(
            "Ignoring stale BuoyIndex announce: gen {} < current gen {}",
            generation,
            prev.generation());
        return;
      }
      if (generation == prev.generation()) {
        if (!versionHash.equals(prev.hash())) {
          LOG.warn(
              "BuoyIndex generation {} announced with conflicting hashes ({} vs {}); keeping the"
                  + " first and ignoring the later — the generation source is not monotonic",
              generation,
              prev.hash(),
              versionHash);
        }
        return; // same generation → idempotent no-op (no event)
      }
    } while (!clusterVersion.compareAndSet(prev, next));
    LOG.debug(
        "BuoyIndex version updated: gen {} → {} ({})", prev.generation(), generation, versionHash);
    fireEvent(new MembershipEvent.BuoyIndexUpdated(versionHash));
  }

  /**
   * Returns the current cluster-wide BuoyIndex version hash, or an empty string if no version has
   * been announced yet.
   */
  public String currentVersionHash() {
    return clusterVersion.get().hash();
  }

  /**
   * Returns the current cluster-wide BuoyIndex generation, or {@link Long#MIN_VALUE} if no version
   * has been announced yet.
   */
  public long currentGeneration() {
    return clusterVersion.get().generation();
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
