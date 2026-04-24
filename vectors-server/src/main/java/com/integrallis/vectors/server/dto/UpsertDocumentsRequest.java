package com.integrallis.vectors.server.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Inbound body for {@code POST /v1/collections/{name}/documents}. The {@code documents} array is
 * upserted in order; any dimension mismatch, duplicate id, or missing vector aborts the batch and
 * yields a {@code 400 Bad Request} without committing the generation.
 */
public record UpsertDocumentsRequest(List<DocumentDto> documents) {

  @JsonCreator
  public UpsertDocumentsRequest(@JsonProperty("documents") List<DocumentDto> documents) {
    this.documents = documents == null ? List.of() : List.copyOf(documents);
  }
}
