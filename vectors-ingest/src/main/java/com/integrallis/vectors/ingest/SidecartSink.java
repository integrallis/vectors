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
 * SPI for the optional row-store companion of one ingestion pipeline (text/blob columns keyed by
 * document id). Implementations must be idempotent on id (e.g. {@code MERGE INTO} / {@code INSERT
 * OR REPLACE}) so a replayed batch produces the same final state.
 */
public interface SidecartSink extends AutoCloseable {

  /** Writes every doc in {@code batch} to the sidecart, idempotently by id. */
  void writeAll(Batch batch) throws IOException;

  /** Total number of rows that have been successfully written through this sink. */
  long writtenCount();

  @Override
  default void close() throws IOException {}
}
