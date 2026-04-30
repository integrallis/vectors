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
import java.time.Instant;

/**
 * Collection metadata returned by the server. Mirrors the server-side {@code
 * com.integrallis.vectors.server.dto.CollectionInfo} wire format.
 *
 * @param name URL-safe collection name
 * @param dimension fixed vector dimension
 * @param metric similarity function (e.g. {@code COSINE})
 * @param indexType index backend (e.g. {@code HNSW})
 * @param quantizer quantizer kind (e.g. {@code NONE})
 * @param size number of live documents currently committed
 * @param createdAt UTC creation instant; may be null when unknown
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record CollectionInfo(
    String name,
    int dimension,
    String metric,
    String indexType,
    String quantizer,
    long size,
    Instant createdAt) {}
