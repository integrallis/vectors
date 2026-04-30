/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.server.client;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

/**
 * Response from {@code POST /v1/collections/{name}/vectors-batch}.
 *
 * @param vectors dense matrix of resolved vectors (rows == requested ids minus missing)
 * @param missing ids that did not resolve to a live document
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record VectorsBatchResponse(float[][] vectors, List<String> missing) {}
