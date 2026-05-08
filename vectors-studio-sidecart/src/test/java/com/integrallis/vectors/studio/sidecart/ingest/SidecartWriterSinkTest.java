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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.ingest.Batch;
import com.integrallis.vectors.ingest.EmbeddedDoc;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartSource;
import com.integrallis.vectors.studio.sidecart.sources.H2SidecartWriter;
import com.integrallis.vectors.studio.sidecart.sources.SidecartUpsert;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SidecartWriterSinkTest {

  private static EmbeddedDoc d(long offset, String id, String text) {
    return new EmbeddedDoc(IngestDoc.text(id, text), new float[] {1f, 2f}, offset);
  }

  @Test
  void forwardsBatchAsUpserts() throws Exception {
    AtomicInteger calls = new AtomicInteger();
    List<SidecartUpsert> received = new ArrayList<>();
    SidecartWriterSink.RowWriter stub =
        rows -> {
          calls.incrementAndGet();
          rows.forEach(received::add);
        };
    try (SidecartWriterSink sink = new SidecartWriterSink(stub)) {
      Batch b = new Batch(0, List.of(d(0, "a", "alpha"), d(1, "b", "beta")));
      sink.writeAll(b);
      assertThat(sink.writtenCount()).isEqualTo(2L);
    }
    assertThat(calls.get()).isEqualTo(1);
    assertThat(received).extracting(SidecartUpsert::id).containsExactly("a", "b");
    assertThat(received).extracting(SidecartUpsert::text).containsExactly("alpha", "beta");
  }

  @Test
  void preservesBlobAndMime() throws Exception {
    List<SidecartUpsert> received = new ArrayList<>();
    SidecartWriterSink.RowWriter stub = rows -> rows.forEach(received::add);
    Batch b =
        new Batch(
            0,
            List.of(
                new EmbeddedDoc(
                    new IngestDoc(
                        "blob", null, new byte[] {1, 2, 3}, "application/octet-stream", null, null),
                    new float[] {1f},
                    0L)));
    try (SidecartWriterSink sink = new SidecartWriterSink(stub)) {
      sink.writeAll(b);
    }
    assertThat(received).hasSize(1);
    SidecartUpsert row = received.get(0);
    assertThat(row.id()).isEqualTo("blob");
    assertThat(row.text()).isNull();
    assertThat(row.blob()).containsExactly(1, 2, 3);
    assertThat(row.mime()).isEqualTo("application/octet-stream");
  }

  @Test
  void multipleBatchesAccumulateCount() throws Exception {
    SidecartWriterSink.RowWriter stub = rows -> {};
    try (SidecartWriterSink sink = new SidecartWriterSink(stub)) {
      sink.writeAll(new Batch(0, List.of(d(0, "a", "x"))));
      sink.writeAll(new Batch(1, List.of(d(1, "b", "y"), d(2, "c", "z"))));
      assertThat(sink.writtenCount()).isEqualTo(3L);
    }
  }

  @Test
  void closeForwardsToWriter() throws Exception {
    AtomicInteger closed = new AtomicInteger();
    SidecartWriterSink.RowWriter stub =
        new SidecartWriterSink.RowWriter() {
          @Override
          public void putAll(Iterable<SidecartUpsert> rows) {}

          @Override
          public void close() {
            closed.incrementAndGet();
          }
        };
    SidecartWriterSink sink = new SidecartWriterSink(stub);
    sink.close();
    assertThat(closed.get()).isEqualTo(1);
  }

  @Test
  void h2AdapterEndToEnd() throws Exception {
    String url = "jdbc:h2:mem:test_" + UUID.randomUUID() + ";DB_CLOSE_DELAY=-1";
    H2SidecartWriter writer = new H2SidecartWriter(url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    writer.ensureSchema();
    SidecartWriterSink sink = SidecartWriterSink.forH2(writer);
    sink.writeAll(new Batch(0, List.of(d(0, "a", "alpha doc"), d(1, "b", "beta doc"))));
    sink.writeAll(new Batch(1, List.of(d(2, "c", "gamma doc"))));
    assertThat(sink.writtenCount()).isEqualTo(3L);

    H2SidecartSource source = new H2SidecartSource(url, "sa", "", "DOCS", "ID", "TEXT", null, null);
    assertThat(source.get("a")).isPresent().get().extracting("text").isEqualTo("alpha doc");
    assertThat(source.get("c")).isPresent().get().extracting("text").isEqualTo("gamma doc");
    assertThat(source.get("missing")).isEmpty();
    sink.close();
  }
}
