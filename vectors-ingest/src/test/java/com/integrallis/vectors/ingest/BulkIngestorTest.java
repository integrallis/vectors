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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.embedders.PrecomputedEmbedder;
import com.integrallis.vectors.ingest.sinks.NoopSidecartSink;
import com.integrallis.vectors.ingest.sinks.NoopVectorSink;
import com.integrallis.vectors.ingest.sources.IterableSource;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BulkIngestorTest {

  private static RetryPolicy noRetry() {
    return new RetryPolicy(
        1, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
  }

  @Test
  void smokeIngestEndToEnd() throws Exception {
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var cursor = new InMemoryCursor();
    List<IngestDoc> docs = new ArrayList<>();
    for (int i = 0; i < 12; i++) docs.add(IngestDoc.text("id-" + i, "doc-" + i));
    try (BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(vs)
            .sidecartSink(ss)
            .embedder(new CapturingSinks.FakeEmbedder(3))
            .batchPolicy(new BatchPolicy(5, 1024L * 1024L, Duration.ofSeconds(5)))
            .cursor(cursor)
            .retryPolicy(noRetry())
            .embeddingConcurrency(2)
            .queueCapacity(8)
            .build()) {
      IngestResult r = ing.ingest(IterableSource.of("smoke", docs));
      assertThat(r.docsCommitted()).isEqualTo(12L);
      assertThat(r.batchesCommitted()).isEqualTo(3L); // 5+5+2
      assertThat(r.lastCursor()).isEqualTo(11L);
    }
    assertThat(vs.committedCount()).isEqualTo(12L);
    assertThat(ss.writtenCount()).isEqualTo(12L);
    assertThat(cursor.load("smoke")).isEqualTo(11L);
  }

  @Test
  void buildRequiresVectorSink() {
    assertThatThrownBy(
            () -> BulkIngestor.builder().embedder(new CapturingSinks.FakeEmbedder(2)).build())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("vectorSink");
  }

  @Test
  void buildRequiresEmbedder() {
    assertThatThrownBy(() -> BulkIngestor.builder().vectorSink(new NoopVectorSink()).build())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("embedder");
  }

  @Test
  void builderRejectsNonPositiveTuning() {
    BulkIngestor.Builder b =
        BulkIngestor.builder()
            .vectorSink(new NoopVectorSink())
            .embedder(new PrecomputedEmbedder(2));
    assertThatThrownBy(() -> b.embeddingConcurrency(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> b.queueCapacity(-1)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void closedIngestorRefusesIngest() throws Exception {
    BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(new NoopVectorSink())
            .sidecartSink(new NoopSidecartSink())
            .embedder(new PrecomputedEmbedder(2))
            .build();
    ing.close();
    assertThatThrownBy(() -> ing.ingest(IterableSource.copyOf("e", List.of())))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("closed");
  }

  @Test
  void metricsReturnZeroBeforeFirstIngest() throws Exception {
    try (BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(new NoopVectorSink())
            .embedder(new PrecomputedEmbedder(2))
            .queueCapacity(16)
            .build()) {
      IngestMetrics m = ing.metrics();
      assertThat(m.docsRead()).isZero();
      assertThat(m.docsCommitted()).isZero();
      assertThat(m.queueCapacity()).isEqualTo(16);
    }
  }

  @Test
  void metricsAfterIngestReflectCounters() throws Exception {
    List<IngestDoc> docs = new ArrayList<>();
    for (int i = 0; i < 4; i++) {
      docs.add(new IngestDoc("p-" + i, "x", null, "text/plain", null, new float[] {1f, 2f}));
    }
    try (BulkIngestor ing =
        BulkIngestor.builder()
            .vectorSink(new NoopVectorSink())
            .embedder(new PrecomputedEmbedder(2))
            .batchPolicy(new BatchPolicy(2, 1024L, Duration.ofSeconds(5)))
            .retryPolicy(noRetry())
            .build()) {
      ing.ingest(IterableSource.of("m", docs));
      IngestMetrics m = ing.metrics();
      assertThat(m.docsRead()).isEqualTo(4L);
      assertThat(m.docsCommitted()).isEqualTo(4L);
      assertThat(m.batchesCommitted()).isEqualTo(2L);
    }
  }
}
