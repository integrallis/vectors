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
import java.util.Objects;

/**
 * Controls when the {@link BatchAccumulator} flushes a {@link Batch}. A flush is triggered by the
 * first of three independent thresholds: doc count, accumulated payload size, or wall time since
 * the first doc was added.
 *
 * @param maxDocs maximum docs per batch (must be &gt; 0)
 * @param maxBytes maximum accumulated text+blob bytes per batch (must be &gt; 0)
 * @param maxLatency maximum age of an open batch (must be positive)
 */
public record BatchPolicy(int maxDocs, long maxBytes, Duration maxLatency) {

  /** Default policy: 1024 docs, 32 MiB, 1 s latency. */
  public static BatchPolicy defaults() {
    return new BatchPolicy(1024, 32L * 1024L * 1024L, Duration.ofSeconds(1));
  }

  public BatchPolicy {
    if (maxDocs <= 0) {
      throw new IllegalArgumentException("maxDocs must be > 0");
    }
    if (maxBytes <= 0) {
      throw new IllegalArgumentException("maxBytes must be > 0");
    }
    Objects.requireNonNull(maxLatency, "maxLatency");
    if (maxLatency.isZero() || maxLatency.isNegative()) {
      throw new IllegalArgumentException("maxLatency must be positive");
    }
  }
}
