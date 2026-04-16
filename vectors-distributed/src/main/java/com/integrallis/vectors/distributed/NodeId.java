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
