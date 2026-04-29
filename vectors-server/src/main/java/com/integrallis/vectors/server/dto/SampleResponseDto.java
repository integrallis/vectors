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
import java.util.List;

/**
 * Response body for {@code GET /v1/collections/{name}/sample?n=}.
 *
 * <p>Returns a deterministic-but-effectively-random subset of {@code n} live documents. {@code
 * metadata} is omitted when {@code includeMetadata=false}.
 *
 * @param ids sampled document ids
 * @param vectors sampled vectors, one per id, in the same order
 * @param metadata optional list of per-document metadata as JSON nodes; null when not requested
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record SampleResponseDto(String[] ids, float[][] vectors, List<JsonNode> metadata) {}
