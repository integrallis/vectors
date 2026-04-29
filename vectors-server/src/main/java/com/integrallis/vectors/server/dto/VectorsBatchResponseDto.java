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
 * Response body for {@code POST /v1/collections/{name}/vectors-batch}.
 *
 * <p>{@code vectors[i]} is the vector for the request id at the same index. Ids that did not
 * resolve to a live document are listed in {@code missing} and are absent from {@code vectors} (the
 * response arrays are dense).
 *
 * @param vectors dense matrix of resolved vectors (rows == ids minus missing)
 * @param missing ids that did not resolve in the live generation
 */
public record VectorsBatchResponseDto(float[][] vectors, List<String> missing) {}
