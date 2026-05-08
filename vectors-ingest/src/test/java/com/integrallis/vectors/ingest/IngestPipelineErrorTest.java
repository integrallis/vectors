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
import com.integrallis.vectors.ingest.sources.IterableSource;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IngestPipelineErrorTest {

  private static IngestSource source(int n) {
    List<IngestDoc> docs = new ArrayList<>(n);
    for (int i = 0; i < n; i++) docs.add(IngestDoc.text("id-" + i, "x"));
    return IterableSource.of("test", docs);
  }

  private static RetryPolicy noRetry() {
    return new RetryPolicy(
        1, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
  }

  @Test
  void sinkFailureFailFastBubblesIOException() {
    var vs = new CapturingSinks.CapturingVectorSink();
    vs.failOnNextAddAll = true;
    var ss = new CapturingSinks.CapturingSidecartSink();
    BatchPolicy bp = new BatchPolicy(2, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        new IngestPipeline(
            source(4),
            new CapturingSinks.FakeEmbedder(2),
            vs,
            ss,
            new InMemoryCursor(),
            bp,
            noRetry(),
            ErrorHandler.failFast(),
            2,
            16);
    assertThatThrownBy(p::run)
        .isInstanceOfAny(IOException.class, RuntimeException.class)
        .hasMessageContaining("simulated addAll failure");
  }

  @Test
  void sinkFailureContinueRecordsErrorButProcessesRemaining() throws Exception {
    var vs = new CapturingSinks.CapturingVectorSink();
    vs.failOnNextAddAll = true;
    var ss = new CapturingSinks.CapturingSidecartSink();
    BatchPolicy bp = new BatchPolicy(2, 1024L * 1024L, Duration.ofSeconds(5));
    AtomicReference<ErrorHandler.IngestErrorContext> seen = new AtomicReference<>();
    ErrorHandler eh = ErrorHandler.continueOnError(seen::set);
    IngestPipeline p =
        new IngestPipeline(
            source(6),
            new CapturingSinks.FakeEmbedder(2),
            vs,
            ss,
            new InMemoryCursor(),
            bp,
            noRetry(),
            eh,
            2,
            16);
    IngestResult r = p.run();
    assertThat(seen.get()).isNotNull();
    assertThat(seen.get().stage()).isEqualTo("commit");
    // 6 docs / batch=2 → 3 batches; first fails, the rest commit.
    assertThat(r.batchesCommitted()).isEqualTo(2L);
    assertThat(r.docsCommitted()).isEqualTo(4L);
    assertThat(r.firstError()).isPresent();
  }

  @Test
  void emptySourceProducesEmptyResult() throws Exception {
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    BatchPolicy bp = new BatchPolicy(8, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        new IngestPipeline(
            IterableSource.copyOf("empty", List.of()),
            new CapturingSinks.FakeEmbedder(2),
            vs,
            ss,
            new InMemoryCursor(),
            bp,
            noRetry(),
            ErrorHandler.failFast(),
            2,
            16);
    IngestResult r = p.run();
    assertThat(r.docsRead()).isZero();
    assertThat(r.docsCommitted()).isZero();
    assertThat(r.batchesCommitted()).isZero();
    assertThat(r.firstError()).isEmpty();
  }
}
