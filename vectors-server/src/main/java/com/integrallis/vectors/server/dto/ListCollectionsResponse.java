package com.integrallis.vectors.server.dto;

import java.util.List;

/**
 * Outbound body for {@code GET /v1/collections}.
 *
 * @param collections unmodifiable snapshot of all currently-registered collections
 */
public record ListCollectionsResponse(List<CollectionInfo> collections) {

  /** Defensive copy to keep the record truly immutable on the wire boundary. */
  public ListCollectionsResponse {
    collections = List.copyOf(collections);
  }
}
