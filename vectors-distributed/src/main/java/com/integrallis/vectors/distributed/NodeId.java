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

import java.util.Objects;

/**
 * Opaque identifier for a cluster node. Immutable value type.
 *
 * @param id the string identifier (must not be null or blank)
 */
public record NodeId(String id) {

  public NodeId {
    Objects.requireNonNull(id, "id must not be null");
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
  }

  @Override
  public String toString() {
    return "NodeId[" + id + "]";
  }
}
