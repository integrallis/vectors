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
