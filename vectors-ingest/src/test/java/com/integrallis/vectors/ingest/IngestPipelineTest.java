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

import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.sources.IterableSource;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IngestPipelineTest {

  private static IngestSource source(int n) {
    List<IngestDoc> docs = new ArrayList<>(n);
    for (int i = 0; i < n; i++) docs.add(IngestDoc.text("id-" + i, "text-" + i));
    return IterableSource.of("test", docs);
  }

  private static RetryPolicy noRetry() {
    return new RetryPolicy(
        1, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
  }

  private static IngestPipeline pipeline(
      IngestSource src,
      Embedder e,
      VectorSink vs,
      SidecartSink ss,
      IngestCursor c,
      BatchPolicy bp,
      ErrorHandler eh) {
    return new IngestPipeline(src, e, vs, ss, c, bp, noRetry(), eh, 2, 32);
  }

  @Test
  void singleBatchHappyPath() throws Exception {
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(3);
    var cursor = new InMemoryCursor();
    BatchPolicy bp = new BatchPolicy(100, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p = pipeline(source(10), emb, vs, ss, cursor, bp, ErrorHandler.failFast());
    IngestResult r = p.run();
    assertThat(r.docsRead()).isEqualTo(10L);
    assertThat(r.docsCommitted()).isEqualTo(10L);
    assertThat(r.batchesCommitted()).isEqualTo(1L);
    assertThat(r.lastCursor()).isEqualTo(9L);
    assertThat(r.firstError()).isEmpty();
    assertThat(vs.committed).hasSize(1);
    assertThat(ss.received).hasSize(1);
    assertThat(cursor.load("test")).isEqualTo(9L);
  }

  @Test
  void multipleBatchesPreserveSourceOrder() throws Exception {
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(2);
    BatchPolicy bp = new BatchPolicy(7, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(source(50), emb, vs, ss, new InMemoryCursor(), bp, ErrorHandler.failFast());
    IngestResult r = p.run();
    assertThat(r.docsCommitted()).isEqualTo(50L);
    assertThat(r.batchesCommitted()).isEqualTo(8L); // 7*7=49 + 1 partial
    // verify ids appear in source order across all committed batches
    List<String> seen = new ArrayList<>();
    for (Batch b : vs.committed) for (EmbeddedDoc d : b.docs()) seen.add(d.doc().id());
    List<String> expected = new ArrayList<>();
    for (int i = 0; i < 50; i++) expected.add("id-" + i);
    assertThat(seen).isEqualTo(expected);
    // monotonic offsets
    long prev = -1;
    for (Batch b : vs.committed) {
      assertThat(b.lastSourceOffset()).isGreaterThan(prev);
      prev = b.lastSourceOffset();
    }
    assertThat(r.lastCursor()).isEqualTo(49L);
  }

  @Test
  void honoursSourceStartOffset() throws Exception {
    List<IngestDoc> docs = new ArrayList<>();
    for (int i = 0; i < 10; i++) docs.add(IngestDoc.text("id-" + i, "x"));
    IngestSource src = IterableSource.resuming("resume", docs, 4);
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    BatchPolicy bp = new BatchPolicy(100, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(
            src,
            new CapturingSinks.FakeEmbedder(2),
            vs,
            ss,
            new InMemoryCursor(),
            bp,
            ErrorHandler.failFast());
    IngestResult r = p.run();
    assertThat(r.docsCommitted()).isEqualTo(6L);
    assertThat(r.lastCursor()).isEqualTo(9L);
    assertThat(vs.committed.get(0).docs().get(0).sourceOffset()).isEqualTo(4L);
  }

  @Test
  void embedderFailureSurfacesAndAbortsRun() {
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(2);
    emb.failNext = true;
    BatchPolicy bp = new BatchPolicy(4, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(source(8), emb, vs, ss, new InMemoryCursor(), bp, ErrorHandler.failFast());
    // The producer's embed failure surfaces via firstError; depending on timing the consumer may
    // also encounter an empty stream → we accept either a thrown exception or a populated
    // firstError.
    try {
      IngestResult r = p.run();
      assertThat(r.firstError()).isPresent();
    } catch (Exception e) {
      assertThat(e).hasMessageContaining("simulated embed failure");
    }
  }
}
