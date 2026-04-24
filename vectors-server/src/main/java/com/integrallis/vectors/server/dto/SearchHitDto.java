package com.integrallis.vectors.server.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import com.integrallis.vectors.db.SearchResult;

/**
 * Outbound hit for {@code POST /v1/collections/{name}/search}. Mirrors {@link
 * com.integrallis.vectors.db.SearchResult.Hit} on the wire.
 *
 * <p>{@code vector}, {@code text}, and {@code metadata} are projected in accordance with the
 * originating {@link SearchQuery#includeVector()}/{@code includeText}/{@code includeMetadata} flags
 * and omitted from the JSON payload when null (@JsonInclude NON_NULL).
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record SearchHitDto(String id, float score, float[] vector, String text, JsonNode metadata) {

  /**
   * Projects a vectors-db hit into the wire form.
   *
   * @param hit the facade hit
   * @param includeVector whether to emit the stored vector
   * @param includeText whether to emit the stored text
   * @param includeMetadata whether to emit the stored metadata
   */
  public static SearchHitDto from(
      SearchResult.Hit hit, boolean includeVector, boolean includeText, boolean includeMetadata) {
    float[] v = includeVector ? hit.document().vector() : null;
    String t = includeText ? hit.document().text() : null;
    JsonNode md = includeMetadata ? MetadataCodec.toJson(hit.document().metadata()) : null;
    return new SearchHitDto(hit.id(), hit.score(), v, t, md);
  }
}
