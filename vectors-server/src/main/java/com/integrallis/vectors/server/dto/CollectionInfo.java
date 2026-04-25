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

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionConfig;
import java.time.Instant;

/**
 * Outbound body for {@code GET /v1/collections/{name}} and {@code POST /v1/collections}.
 *
 * <p>Enum-valued fields are serialized as strings (e.g. {@code "COSINE"}, {@code "HNSW"}, {@code
 * "NONE"}) so the wire format stays stable across Java enum refactors.
 *
 * @param name URL-safe collection name
 * @param dimension fixed vector dimension
 * @param metric similarity function (e.g. {@code COSINE})
 * @param indexType index backend (e.g. {@code HNSW})
 * @param quantizer quantizer kind (e.g. {@code NONE})
 * @param size number of live documents currently committed
 * @param createdAt UTC creation instant (ISO-8601 via {@code JavaTimeModule})
 */
public record CollectionInfo(
    String name,
    int dimension,
    String metric,
    String indexType,
    String quantizer,
    int size,
    Instant createdAt) {

  /**
   * Projects a live {@link VectorCollection} plus a registry-tracked {@code createdAt} into the
   * outbound wire form.
   */
  public static CollectionInfo of(String name, VectorCollection collection, Instant createdAt) {
    VectorCollectionConfig config = collection.config();
    return new CollectionInfo(
        name,
        config.dimension(),
        config.metric().name(),
        config.indexType().name(),
        config.quantizerKind().name(),
        collection.size(),
        createdAt);
  }
}
