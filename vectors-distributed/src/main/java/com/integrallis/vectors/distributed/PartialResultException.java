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

/**
 * Thrown by cluster-wide aggregates (e.g. {@link DistributedVectorCollection#size()} /
 * {@link DistributedVectorCollection#physicalSize()}) when one or more nodes could not be reached
 * within the configured timeout or threw while responding.
 *
 * <p>Unlike {@code search}, which silently returns partial ranked results, a partial <em>count</em>
 * is misleading — a caller comparing sizes before/after an ingest could mistake a briefly
 * unreachable node for data loss. So the aggregate fails fast rather than lying, but still carries
 * the {@link #partialTotal()} it did gather and the {@link #unreachableNodes()} that were missing,
 * letting a caller that is willing to accept a partial answer recover it. The call is always bounded
 * by the timeout, so it never hangs on a slow node.
 */
public final class PartialResultException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  private final transient int partialTotal;
  private final transient Set<NodeId> unreachableNodes;

  PartialResultException(String message, int partialTotal, Set<NodeId> unreachableNodes) {
    super(message);
    this.partialTotal = partialTotal;
    this.unreachableNodes = Set.copyOf(unreachableNodes);
  }

  /** The aggregate over only the nodes that responded successfully. */
  public int partialTotal() {
    return partialTotal;
  }

  /** The nodes that timed out or threw, and were therefore excluded from {@link #partialTotal()}. */
  public Set<NodeId> unreachableNodes() {
    return unreachableNodes;
  }
}
