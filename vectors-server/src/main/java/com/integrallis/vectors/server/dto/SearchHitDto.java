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
    return from(hit, hit.score(), includeVector, includeText, includeMetadata);
  }

  /**
   * Projects a vectors-db hit into the wire form with a caller-supplied score, used when a route
   * returns a score from a fusion strategy instead of the underlying vector search.
   */
  public static SearchHitDto from(
      SearchResult.Hit hit,
      float score,
      boolean includeVector,
      boolean includeText,
      boolean includeMetadata) {
    float[] v = includeVector ? hit.document().vector() : null;
    String t = includeText ? hit.document().text() : null;
    JsonNode md = includeMetadata ? MetadataCodec.toJson(hit.document().metadata()) : null;
    return new SearchHitDto(hit.id(), score, v, t, md);
  }
}
