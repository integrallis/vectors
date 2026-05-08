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

import java.util.Objects;

/**
 * Internal pairing of an {@link IngestDoc} with its embedding vector and the source-relative offset
 * that produced it. The {@code sourceOffset} is what the {@link IngestCursor} ultimately persists,
 * so it must be monotonic in the order the source emitted docs.
 */
public record EmbeddedDoc(IngestDoc doc, float[] vector, long sourceOffset) {

  public EmbeddedDoc {
    Objects.requireNonNull(doc, "doc");
    Objects.requireNonNull(vector, "vector");
    if (vector.length == 0) {
      throw new IllegalArgumentException("vector must be non-empty");
    }
    if (sourceOffset < 0) {
      throw new IllegalArgumentException("sourceOffset must be >= 0");
    }
  }
}
