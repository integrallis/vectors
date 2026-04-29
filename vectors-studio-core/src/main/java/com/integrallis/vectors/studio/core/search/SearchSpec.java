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
package com.integrallis.vectors.studio.core.search;

import java.util.Map;

/**
 * UI-side search request. Either {@code queryVector} or {@code queryText} must be supplied (both
 * may be present for hybrid search).
 *
 * @param queryVector dense vector query, or null for text-only search
 * @param queryText raw text query, or null for vector-only search
 * @param k number of hits to return
 * @param filter opaque filter map mirroring the server-side AST; null for no filter
 * @param includeVector whether each hit should carry its vector
 * @param includeText whether each hit should carry its text
 * @param includeMetadata whether each hit should carry its metadata
 */
public record SearchSpec(
    float[] queryVector,
    String queryText,
    int k,
    Map<String, Object> filter,
    boolean includeVector,
    boolean includeText,
    boolean includeMetadata) {}
