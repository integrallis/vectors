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
package com.integrallis.vectors.studio.sidecart.ingest;

import com.integrallis.vectors.ingest.Batch;
import com.integrallis.vectors.ingest.EmbeddedDoc;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.SidecartSink;
import com.integrallis.vectors.studio.sidecart.sources.D1SidecartWriter;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartWriter;
import com.integrallis.vectors.studio.sidecart.sources.SidecartUpsert;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * {@link SidecartSink} adapter that forwards each ingest {@link Batch} to a row-level sidecart
 * writer. Concrete writers are passed in via a small {@link RowWriter} abstraction so the same sink
 * can drive an {@link H2SidecartWriter}, a {@link D1SidecartWriter}, or any future adapter without
 * leaking those types into {@code vectors-ingest}.
 *
 * <p>Both shipped writers are idempotent on {@code id} (H2 {@code MERGE INTO}, D1 {@code INSERT OR
 * REPLACE}), so replays caused by a cursor rollback or a retry are safe.
 */
public final class SidecartWriterSink implements SidecartSink {

  /**
   * Minimal row-writer surface implemented by every backend. Kept package-friendly so callers can
   * construct a sink against any writer that exposes a batched-upsert path.
   */
  public interface RowWriter extends AutoCloseable {
    /** Upserts every row in {@code rows}. Implementations are expected to be idempotent on id. */
    void putAll(Iterable<SidecartUpsert> rows);

    @Override
    default void close() {}
  }

  private final RowWriter writer;
  private final AtomicLong written = new AtomicLong();

  public SidecartWriterSink(RowWriter writer) {
    this.writer = Objects.requireNonNull(writer, "writer");
  }

  /** Adapts an {@link H2SidecartWriter}; assumes {@code ensureSchema()} has been invoked. */
  public static SidecartWriterSink forH2(H2SidecartWriter writer) {
    Objects.requireNonNull(writer, "writer");
    return new SidecartWriterSink(
        new RowWriter() {
          @Override
          public void putAll(Iterable<SidecartUpsert> rows) {
            writer.putAll(rows);
          }

          @Override
          public void close() {
            writer.close();
          }
        });
  }

  /** Adapts a {@link D1SidecartWriter}; assumes {@code ensureSchema()} has been invoked. */
  public static SidecartWriterSink forD1(D1SidecartWriter writer) {
    Objects.requireNonNull(writer, "writer");
    return new SidecartWriterSink(
        new RowWriter() {
          @Override
          public void putAll(Iterable<SidecartUpsert> rows) {
            writer.putAll(rows);
          }

          @Override
          public void close() {
            writer.close();
          }
        });
  }

  @Override
  public void writeAll(Batch batch) {
    List<SidecartUpsert> rows = new ArrayList<>(batch.size());
    for (EmbeddedDoc d : batch.docs()) {
      rows.add(toUpsert(d.doc()));
    }
    writer.putAll(rows);
    written.addAndGet(batch.size());
  }

  @Override
  public long writtenCount() {
    return written.get();
  }

  @Override
  public void close() throws IOException {
    try {
      writer.close();
    } catch (Exception e) {
      throw new IOException("sidecart writer close failed", e);
    }
  }

  private static SidecartUpsert toUpsert(IngestDoc d) {
    return new SidecartUpsert(d.id(), d.text(), d.blob(), d.mime());
  }
}
