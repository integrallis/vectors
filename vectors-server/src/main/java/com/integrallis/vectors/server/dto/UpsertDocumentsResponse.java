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

/**
 * Outbound body for {@code POST /v1/collections/{name}/documents}.
 *
 * @param upserted number of documents that were applied and committed
 * @param size total live-document count after the commit
 * @param textIndexed {@code false} when a text index is configured for the collection but its
 *     dual-write failed (the vector write still committed successfully); {@code true} otherwise
 *     (write succeeded, or no text index is configured)
 */
public record UpsertDocumentsResponse(int upserted, int size, boolean textIndexed) {}
