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

import java.util.Optional;

/**
 * Live snapshot of pipeline counters. Returned by {@link BulkIngestor#metrics()} and safe to
 * publish from any thread; it captures point-in-time values and never mutates afterwards.
 */
public record IngestMetrics(
    long docsRead,
    long docsEmbedded,
    long docsCommitted,
    long batchesCommitted,
    long bytesEmbedded,
    int currentQueueDepth,
    int queueCapacity,
    long retryCount,
    Optional<String> lastError) {

  public IngestMetrics {
    if (docsRead < 0
        || docsEmbedded < 0
        || docsCommitted < 0
        || batchesCommitted < 0
        || bytesEmbedded < 0
        || currentQueueDepth < 0
        || queueCapacity < 0
        || retryCount < 0) {
      throw new IllegalArgumentException("counters must be >= 0");
    }
    if (lastError == null) {
      lastError = Optional.empty();
    }
  }
}
