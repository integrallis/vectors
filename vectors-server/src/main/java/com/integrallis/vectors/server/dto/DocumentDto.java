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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.vectors.core.Document;

/**
 * Wire-format document for {@code POST /v1/collections/\u007Bname\u007D/documents} (inbound) and
 * {@code POST /v1/collections/\u007Bname\u007D/search} responses (outbound).
 *
 * <p>When decoded from JSON, {@code vector} is required (the embedded collection rejects null
 * vectors on insert). On the outbound path {@code vector} is only emitted when the search request
 * asked for it via {@code includeVector}.
 *
 * @param id external document identifier
 * @param vector embedding, 1-D float array; required on insert, nullable on projection
 * @param text optional raw text; nullable
 * @param metadata plain JSON object of scalar values or string arrays; nullable or missing means
 *     empty
 * @param blob optional Base64-encoded binary data (e.g. image bytes); nullable
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record DocumentDto(String id, float[] vector, String text, JsonNode metadata, String blob) {

  @JsonCreator
  public DocumentDto(
      @JsonProperty("id") String id,
      @JsonProperty("vector") float[] vector,
      @JsonProperty("text") String text,
      @JsonProperty("metadata") JsonNode metadata,
      @JsonProperty("blob") String blob) {
    this.id = id;
    this.vector = vector;
    this.text = text;
    this.metadata = metadata;
    this.blob = blob;
  }

  /** Validates the minimal fields required to stage an insert and builds a {@link Document}. */
  public Document toDocument() {
    if (id == null || id.isEmpty()) {
      throw new IllegalArgumentException("document id is required");
    }
    if (vector == null) {
      throw new IllegalArgumentException("document vector is required for id='" + id + "'");
    }
    return new Document(id, vector, text, MetadataCodec.fromJson(metadata));
  }

  /**
   * Projects a {@link Document} into a wire DTO. {@code includeVector} / {@code includeText} /
   * {@code includeMetadata} mirror the flags on the originating {@link
   * com.integrallis.vectors.db.SearchRequest} and control which fields are emitted on the wire.
   */
  public static DocumentDto from(
      Document doc, boolean includeVector, boolean includeText, boolean includeMetadata) {
    float[] v = includeVector ? doc.vector() : null;
    String t = includeText ? doc.text() : null;
    ObjectNode md = includeMetadata ? MetadataCodec.toJson(doc.metadata()) : null;
    return new DocumentDto(doc.id(), v, t, md, null);
  }
}
