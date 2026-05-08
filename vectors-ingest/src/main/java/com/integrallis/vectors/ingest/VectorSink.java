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
 * SPI for the durable vector destination of one ingestion pipeline. {@link #addAll} stages a batch
 * and {@link #commit} makes it durable — implementations must serialise these calls and provide the
 * atomic batch boundary documented in {@link BulkIngestor}.
 */
public interface VectorSink extends AutoCloseable {

  /** Stages every embedded doc in {@code batch} into the underlying collection. */
  void addAll(Batch batch) throws IOException;

  /** Makes the staged batch durable. */
  void commit() throws IOException;

  /** Total number of docs that have been successfully committed through this sink. */
  long committedCount();

  @Override
  default void close() throws IOException {}
}
