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
