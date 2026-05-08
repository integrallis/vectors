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

/**
 * SPI for a vector embedder. Implementations are expected to be thread-safe — the pipeline calls
 * {@link #embedAll} concurrently from multiple embed workers — and must preserve input order in the
 * returned list.
 */
public interface Embedder {

  /** Short, stable name (used in logs and {@link IngestMetrics}). */
  String name();

  /** Output dimension. Must match the dimension of the destination collection. */
  int dimension();

  /**
   * Batch-friendly embed. Implementations may parallelise internally; the returned list's i-th
   * element must correspond to the i-th input doc.
   */
  List<float[]> embedAll(List<IngestDoc> docs);

  /** Default single-doc convenience that delegates to {@link #embedAll}. */
  default float[] embed(IngestDoc d) {
    return embedAll(List.of(d)).get(0);
  }
}
