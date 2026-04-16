package com.integrallis.vectors.distributed;

import java.util.Set;
import java.util.function.Consumer;

/**
 * Live view of the cluster's node membership and gossip-propagated index version.
 *
 * <p>Implementations are expected to be thread-safe. Change listeners are called on an unspecified
 * thread; listeners must be non-blocking.
 */
public interface ClusterMembership {

  /**
   * Returns the current set of live nodes. The returned set is a snapshot; it does not update as
   * the membership changes.
   *
   * @return unmodifiable snapshot of live node ids (never null)
   */
  Set<NodeId> liveNodes();

  /**
   * Registers a listener that will be notified of membership and index-version changes.
   *
   * @param listener the listener to register (must be non-blocking)
   */
  void registerChangeListener(Consumer<MembershipEvent> listener);

  /** Sealed hierarchy of membership change events. */
  sealed interface MembershipEvent
      permits MembershipEvent.NodeJoined,
          MembershipEvent.NodeLeft,
          MembershipEvent.BuoyIndexUpdated {

    /** A new node has joined the cluster. */
    record NodeJoined(NodeId nodeId) implements MembershipEvent {}

    /** A node has left or become unreachable. */
    record NodeLeft(NodeId nodeId) implements MembershipEvent {}

    /**
     * The BuoyIndex (centroid routing index) has been updated across the cluster. Receivers should
     * reload the index from the shared store.
     */
    record BuoyIndexUpdated(String newVersionHash) implements MembershipEvent {}
  }
}
