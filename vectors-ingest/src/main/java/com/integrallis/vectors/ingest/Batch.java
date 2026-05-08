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

import java.util.List;
import java.util.Objects;

/**
 * A monotonically-numbered, ordered collection of {@link EmbeddedDoc}s flushed by the {@link
 * BatchAccumulator} into the sinks atomically.
 *
 * @param batchId monotonic identifier (0, 1, 2, …) assigned by the pipeline
 * @param docs immutable list of docs in source order
 */
public record Batch(long batchId, List<EmbeddedDoc> docs) {

  public Batch {
    if (batchId < 0) {
      throw new IllegalArgumentException("batchId must be >= 0");
    }
    Objects.requireNonNull(docs, "docs");
    docs = List.copyOf(docs);
    if (docs.isEmpty()) {
      throw new IllegalArgumentException("Batch must contain at least one doc");
    }
  }

  /** Number of docs in the batch. */
  public int size() {
    return docs.size();
  }

  /**
   * Returns the source offset of the last doc in this batch (used by the cursor to record the
   * resume point after a successful commit).
   */
  public long lastSourceOffset() {
    return docs.get(docs.size() - 1).sourceOffset();
  }
}
