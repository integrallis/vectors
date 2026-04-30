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
import java.util.Map;

/**
 * A single search result from the vectors-server.
 *
 * @param id the document identifier
 * @param score the similarity score
 * @param vector the stored vector (if requested via {@code includeVector})
 * @param text the stored text (if requested)
 * @param metadata the stored metadata (if requested)
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record SearchHit(
    String id, float score, float[] vector, String text, Map<String, Object> metadata) {}
