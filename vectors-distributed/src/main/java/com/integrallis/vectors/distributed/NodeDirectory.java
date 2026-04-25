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
 * Registry that resolves a {@link NodeId} to its {@link NodeSearchClient}.
 *
 * <p>In-process implementation ({@code InProcessNodeDirectory}) is used for tests. A production
 * implementation would maintain gRPC channels or HTTP clients per node.
 *
 * <p>Implementations must be thread-safe.
 */
public interface NodeDirectory {

  /**
   * Returns the search client for the given node.
   *
   * @param nodeId the target node
   * @return the client (never null)
   * @throws IllegalArgumentException if the node is not registered
   */
  NodeSearchClient clientFor(NodeId nodeId);

  /**
   * Returns the set of all known node ids registered in this directory.
   *
   * @return unmodifiable set (may be empty)
   */
  Set<NodeId> allNodes();
}
