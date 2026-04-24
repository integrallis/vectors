package com.integrallis.vectors.server.dto;

import java.util.List;

/**
 * Outbound body for {@code POST /v1/collections/{name}/search}.
 *
 * @param hits ranked hits, descending by score
 * @param searchTimeNanos end-to-end search latency as reported by the embedded collection
 */
public record SearchResponse(List<SearchHitDto> hits, long searchTimeNanos) {

  public SearchResponse {
    hits = List.copyOf(hits);
  }
}
