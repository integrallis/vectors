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
