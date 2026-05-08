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

import java.time.Duration;
import java.util.Optional;

/**
 * Final outcome of one {@link BulkIngestor#ingest(IngestSource)} call. Counters are cumulative for
 * the source; {@link #firstError} surfaces the earliest non-recoverable failure when the ingestor's
 * error handler chose to continue past it.
 */
public record IngestResult(
    long docsRead,
    long docsEmbedded,
    long docsCommitted,
    long batchesCommitted,
    long bytesEmbedded,
    Duration totalDuration,
    long lastCursor,
    Optional<Throwable> firstError) {

  public IngestResult {
    if (docsRead < 0
        || docsEmbedded < 0
        || docsCommitted < 0
        || batchesCommitted < 0
        || bytesEmbedded < 0
        || lastCursor < 0) {
      throw new IllegalArgumentException("counters must be >= 0");
    }
    if (totalDuration == null || totalDuration.isNegative()) {
      throw new IllegalArgumentException("totalDuration must be non-negative");
    }
    if (firstError == null) {
      firstError = Optional.empty();
    }
  }
}
