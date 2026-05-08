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
package com.integrallis.vectors.ingest;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * One unit of ingestion. Either {@link #text} or {@link #blob} must be present (an embedder may use
 * the {@link #precomputedVector} bypass instead). The record performs defensive copies on
 * construction so callers may mutate the inputs after {@code new IngestDoc(...)} without affecting
 * the stored value.
 *
 * @param id non-blank document identifier
 * @param text optional text payload
 * @param blob optional binary payload (defensively copied)
 * @param mime optional MIME type; sinks infer a default when {@code null}
 * @param attrs optional reserved-for-v2 string attributes (defensively copied to an immutable map)
 * @param precomputedVector optional vector that bypasses the embedder when non-null (defensively
 *     copied)
 */
public record IngestDoc(
    String id,
    String text,
    byte[] blob,
    String mime,
    Map<String, String> attrs,
    float[] precomputedVector) {

  public IngestDoc {
    Objects.requireNonNull(id, "id");
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must be non-blank");
    }
    boolean hasText = text != null && !text.isEmpty();
    boolean hasBlob = blob != null && blob.length > 0;
    boolean hasVec = precomputedVector != null && precomputedVector.length > 0;
    if (!hasText && !hasBlob && !hasVec) {
      throw new IllegalArgumentException(
          "IngestDoc " + id + " requires one of text/blob/precomputedVector");
    }
    blob = (blob == null) ? null : Arrays.copyOf(blob, blob.length);
    precomputedVector =
        (precomputedVector == null)
            ? null
            : Arrays.copyOf(precomputedVector, precomputedVector.length);
    attrs = (attrs == null) ? Map.of() : Collections.unmodifiableMap(new HashMap<>(attrs));
  }

  /** Convenience factory for the common text-only case. */
  public static IngestDoc text(String id, String text) {
    return new IngestDoc(id, text, null, "text/plain", Map.of(), null);
  }

  /** Convenience factory for a doc that already carries its embedding. */
  public static IngestDoc precomputed(String id, String text, float[] vector) {
    return new IngestDoc(id, text, null, "text/plain", Map.of(), vector);
  }

  /** Returns a defensive copy of the blob (or {@code null} if unset). */
  @Override
  public byte[] blob() {
    return blob == null ? null : Arrays.copyOf(blob, blob.length);
  }

  /** Returns a defensive copy of the precomputed vector (or {@code null} if unset). */
  @Override
  public float[] precomputedVector() {
    return precomputedVector == null
        ? null
        : Arrays.copyOf(precomputedVector, precomputedVector.length);
  }
}
