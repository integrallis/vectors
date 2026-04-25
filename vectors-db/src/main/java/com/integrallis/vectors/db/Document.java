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
package com.integrallis.vectors.db;

import java.util.Map;
import java.util.Objects;

/**
 * A document added to a {@link VectorCollection}. Carries an external string id, a float vector,
 * optional text, and typed metadata.
 *
 * <p>The record's canonical constructor copies the metadata map defensively and rejects {@code
 * null} for {@code id}. The {@code vector} is mandatory at insertion time but may be {@code null}
 * in search-result projections when {@code SearchRequest.includeVector} is false. {@code text} and
 * {@code metadata} may also be null — a null metadata map is normalised to an empty map.
 *
 * <p><b>Vector ownership.</b> The {@code vector} array is <i>not</i> defensively copied. The
 * collection stores the reference directly for zero-copy ingestion, and the same reference is
 * handed back through the metadata store and through the backend's vector matrix. Callers must
 * <b>not</b> modify the array after handing it to {@link VectorCollection#add(Document)} — doing so
 * corrupts the stored vector and any subsequent search results. If you reuse a buffer in a
 * batch-insert loop, clone it before constructing each {@link Document}.
 *
 * @param id external identifier (must not be null)
 * @param vector embedding (required on insertion; may be null in projections; stored by reference,
 *     not copied)
 * @param text optional raw text (may be null)
 * @param metadata typed metadata map (null → empty)
 */
public record Document(
    String id, float[] vector, String text, Map<String, MetadataValue> metadata) {

  public Document {
    Objects.requireNonNull(id, "id must not be null");
    metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
  }

  /** Factory for an id + vector only. */
  public static Document of(String id, float[] vector) {
    return new Document(id, vector, null, Map.of());
  }

  /** Factory for an id + vector + text. */
  public static Document of(String id, float[] vector, String text) {
    return new Document(id, vector, text, Map.of());
  }
}
