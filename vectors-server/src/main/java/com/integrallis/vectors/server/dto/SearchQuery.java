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
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * Inbound body for {@code POST /v1/collections/{name}/search}.
 *
 * <p>All fields except {@code queryVector} and {@code k} are optional; unset booleans mean "include
 * everything" (matching the {@link com.integrallis.vectors.db.SearchRequest} builder defaults).
 *
 * @param queryVector the embedding to search for; length must match the collection's dimension
 * @param k number of hits to return (must be positive)
 * @param efSearch optional graph-traversal candidate-list size (HNSW/Vamana only)
 * @param minScore optional score floor; hits below this are dropped from the response
 * @param includeVector whether to emit the stored vector on each hit
 * @param includeText whether to emit the stored text on each hit
 * @param includeMetadata whether to emit the stored metadata on each hit
 * @param filter optional filter AST; Phase 5 grammar, Phase 4 only accepts null
 * @param queryText optional text for hybrid search (combined with vector via fusion)
 * @param hybridMode optional fusion mode: "RRF" (default) or "WEIGHTED"
 */
public record SearchQuery(
    float[] queryVector,
    Integer k,
    Integer efSearch,
    Float minScore,
    Boolean includeVector,
    Boolean includeText,
    Boolean includeMetadata,
    JsonNode filter,
    String queryText,
    String hybridMode) {

  @JsonCreator
  public SearchQuery(
      @JsonProperty("queryVector") float[] queryVector,
      @JsonProperty("k") Integer k,
      @JsonProperty("efSearch") Integer efSearch,
      @JsonProperty("minScore") Float minScore,
      @JsonProperty("includeVector") Boolean includeVector,
      @JsonProperty("includeText") Boolean includeText,
      @JsonProperty("includeMetadata") Boolean includeMetadata,
      @JsonProperty("filter") JsonNode filter,
      @JsonProperty("queryText") String queryText,
      @JsonProperty("hybridMode") String hybridMode) {
    this.queryVector = queryVector;
    this.k = k;
    this.efSearch = efSearch;
    this.minScore = minScore;
    this.includeVector = includeVector;
    this.includeText = includeText;
    this.includeMetadata = includeMetadata;
    this.filter = filter;
    this.queryText = queryText;
    this.hybridMode = hybridMode;
  }

  /** First-field-level validation error, or {@code null} if the request is well-formed. */
  public String validate() {
    if (queryVector == null || queryVector.length == 0) {
      return "queryVector is required and must be non-empty";
    }
    if (k == null || k <= 0) {
      return "k must be a positive integer";
    }
    if (efSearch != null && efSearch <= 0) {
      return "efSearch must be positive";
    }
    return null;
  }

  public boolean includeVectorDefault() {
    return includeVector == null ? true : includeVector;
  }

  public boolean includeTextDefault() {
    return includeText == null ? true : includeText;
  }

  public boolean includeMetadataDefault() {
    return includeMetadata == null ? true : includeMetadata;
  }
}
