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
 * A single hit returned from {@link com.integrallis.vectors.studio.core.connection.StudioBackend}.
 *
 * @param id document id
 * @param score similarity score (semantics depend on the metric)
 * @param vector hit's vector when requested via {@code includeVector}, else null
 * @param text hit's text when requested, else null
 * @param metadata hit's metadata when requested, else null
 */
public record SearchHit(
    String id, double score, float[] vector, String text, Map<String, Object> metadata) {}
