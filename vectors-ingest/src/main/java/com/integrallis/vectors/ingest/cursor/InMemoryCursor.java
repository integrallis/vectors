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
package com.integrallis.vectors.ingest.cursor;

import com.integrallis.vectors.ingest.IngestCursor;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Non-durable {@link IngestCursor} backed by an in-process {@link ConcurrentHashMap}. Useful for
 * tests and for short-lived ingestion runs that don't need to survive a restart.
 */
public final class InMemoryCursor implements IngestCursor {

  private final ConcurrentHashMap<String, Long> offsets = new ConcurrentHashMap<>();

  @Override
  public long load(String sourceName) {
    Objects.requireNonNull(sourceName, "sourceName");
    return offsets.getOrDefault(sourceName, 0L);
  }

  @Override
  public void save(String sourceName, long offset) {
    Objects.requireNonNull(sourceName, "sourceName");
    if (offset < 0) {
      throw new IllegalArgumentException("offset must be >= 0");
    }
    offsets.put(sourceName, offset);
  }
}
