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
package com.integrallis.vectors.core;

import java.util.Map;
import java.util.Objects;

/**
 * A document carrying an external string id, a float vector, optional text, and typed metadata.
 *
 * <p>The record's canonical constructor copies the metadata map defensively and rejects {@code
 * null} for {@code id}. The {@code vector} is mandatory at insertion time but may be {@code null}
 * in search-result projections when {@code SearchRequest.includeVector} is false. {@code text} and
 * {@code metadata} may also be null — a null metadata map is normalised to an empty map.
 *
 * <p><b>Vector ownership.</b> The {@code vector} array on a Document passed to a collection is
 * defensively cloned at the staging boundary, so the caller may safely reuse and mutate their own
 * buffer afterwards (e.g., to batch-insert from a reusable float[]). Conversely, the {@code vector}
 * array on a Document <i>returned</i> by a collection (via {@code search} projections or {@code
 * get}) references the collection's internally-held storage and <b>must not</b> be mutated — doing
 * so corrupts the stored vector and subsequent search results. Treat returned vector arrays as
 * immutable.
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
