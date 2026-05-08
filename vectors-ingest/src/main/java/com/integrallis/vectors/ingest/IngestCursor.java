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

import java.io.IOException;

/**
 * SPI for a durable resume cursor scoped per source name. The pipeline calls {@link #save} only
 * after both the {@link VectorSink} and {@link SidecartSink} commits succeed, so a successfully
 * persisted offset always points to fully-committed downstream state.
 */
public interface IngestCursor {

  /** Returns the offset at which to resume, or {@code 0} when no prior run has been recorded. */
  long load(String sourceName) throws IOException;

  /** Persists {@code offset} as the new resume point for {@code sourceName}. */
  void save(String sourceName, long offset) throws IOException;
}
