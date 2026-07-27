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
  void resumesFromPersistedCursor() throws Exception {
    // Regression (audit ingest #2): commitBatch wrote cursor.save(...) but the pipeline never
    // called
    // cursor.load(), so a restart re-ingested the whole source from offset 0 despite durable
    // progress. With a cursor showing offsets 0..5 already committed, a fresh source must resume at
    // offset 6.
    InMemoryCursor cursor = new InMemoryCursor();
    cursor.save("resume-src", 5L); // last committed 0-based offset

    List<IngestDoc> docs = new ArrayList<>();
    for (int i = 0; i < 10; i++) docs.add(IngestDoc.text("id-" + i, "x"));
    IngestSource src = IterableSource.of("resume-src", docs); // fresh source, its own startOffset 0
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    BatchPolicy bp = new BatchPolicy(100, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(
            src, new CapturingSinks.FakeEmbedder(2), vs, ss, cursor, bp, ErrorHandler.failFast());

    IngestResult r = p.run();

    assertThat(r.docsCommitted()).as("only offsets 6..9 remain").isEqualTo(4L);
    assertThat(r.lastCursor()).isEqualTo(9L);
    assertThat(vs.committed.get(0).docs().get(0).sourceOffset()).isEqualTo(6L);
    assertThat(vs.committed.get(0).docs().get(0).doc().id()).isEqualTo("id-6");
  }

  @Test
  void freshCursorDoesNotSkipAnyDocs() throws Exception {
    // A brand-new (empty) cursor must not be mistaken for "offset 0 committed" — a fresh run
    // ingests
    // every doc from the start.
    InMemoryCursor cursor = new InMemoryCursor();
    IngestPipeline p =
        pipeline(
            source(5),
            new CapturingSinks.FakeEmbedder(2),
            new CapturingSinks.CapturingVectorSink(),
            new CapturingSinks.CapturingSidecartSink(),
            cursor,
            new BatchPolicy(100, 1024L * 1024L, Duration.ofSeconds(5)),
            ErrorHandler.failFast());

    IngestResult r = p.run();

    assertThat(r.docsCommitted()).isEqualTo(5L);
    assertThat(cursor.load("test")).isEqualTo(4L);
  }

  @Test
  void continueOnErrorDoesNotAdvanceCursorPastADroppedBatch() throws Exception {
    // 15 docs, batch size 5 → source offsets 0-4, 5-9, 10-14. The vector sink fails to commit the
    // SECOND batch (5-9); continueOnError drops it and proceeds to the third (10-14).
    var delegate = new CapturingSinks.CapturingVectorSink();
    VectorSink failsSecondCommit =
        new VectorSink() {
          int commits = 0;

          @Override
          public void addAll(Batch batch) throws java.io.IOException {
            delegate.addAll(batch);
          }

          @Override
          public void commit() throws java.io.IOException {
            if (++commits == 2) throw new java.io.IOException("injected commit failure on batch 2");
            delegate.commit();
          }

          @Override
          public long committedCount() {
            return delegate.committedCount();
          }
        };
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(3);
    var cursor = new InMemoryCursor();
    BatchPolicy bp = new BatchPolicy(5, 1024L * 1024L, Duration.ofSeconds(5));

    IngestResult r =
        pipeline(
                source(15),
                emb,
                failsSecondCommit,
                ss,
                cursor,
                bp,
                ErrorHandler.continueOnError(c -> {}))
            .run();

    // Batches 1 and 3 committed (10 docs); batch 2 was dropped.
    assertThat(r.docsCommitted()).isEqualTo(10L);
    // The durable cursor must stay at the last contiguously-committed offset (4) — it must NOT jump
    // to 14 on the back of the third batch, or a restart would skip the dropped 5-9 forever. Before
    // the fix, batch 3 saved cursor = 14 (this assertion failed with 14).
    assertThat(cursor.load("test")).isEqualTo(4L);

    // Resume with a fresh sink + the same cursor: the dropped docs (5-9) are re-processed, not
    // lost.
    var vs2 = new CapturingSinks.CapturingVectorSink();
    pipeline(
            source(15),
            emb,
            vs2,
            new CapturingSinks.CapturingSidecartSink(),
            cursor,
            bp,
            ErrorHandler.failFast())
        .run();
    List<String> reingested =
        vs2.committed.stream().flatMap(b -> b.docs().stream()).map(d -> d.doc().id()).toList();
    assertThat(reingested).contains("id-5", "id-6", "id-7", "id-8", "id-9");
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

  @Test
  void producerClosesSourceIteratorOnHappyPath() throws Exception {
    // Regression: a source whose iterator holds a resource (e.g. JsonlSource's file descriptor) is
    // only self-closed on natural EOF by some implementations. The producer must close it too.
    var src = new TrackingCloseableSource(10);
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(3);
    BatchPolicy bp = new BatchPolicy(4, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(src, emb, vs, ss, new InMemoryCursor(), bp, ErrorHandler.failFast());

    p.run();

    assertThat(src.iteratorClosed()).as("source iterator closed after a clean run").isTrue();
  }

  @Test
  void producerClosesSourceIteratorOnEarlyAbort() {
    // The leak scenario: iteration is abandoned mid-stream because embedding fails. The producer's
    // finally block must still close the iterator, releasing its resources.
    var src = new TrackingCloseableSource(40);
    var vs = new CapturingSinks.CapturingVectorSink();
    var ss = new CapturingSinks.CapturingSidecartSink();
    var emb = new CapturingSinks.FakeEmbedder(2);
    emb.failNext = true;
    BatchPolicy bp = new BatchPolicy(4, 1024L * 1024L, Duration.ofSeconds(5));
    IngestPipeline p =
        pipeline(src, emb, vs, ss, new InMemoryCursor(), bp, ErrorHandler.failFast());

    try {
      p.run();
    } catch (Exception ignored) {
      // abort is expected; the point is the iterator still gets closed
    }

    assertThat(src.iteratorClosed()).as("source iterator closed even on early abort").isTrue();
  }

  /** An {@link IngestSource} whose iterator is {@link java.io.Closeable} and records close(). */
  private static final class TrackingCloseableSource implements IngestSource {
    private final List<IngestDoc> docs;
    private final java.util.concurrent.atomic.AtomicBoolean closed =
        new java.util.concurrent.atomic.AtomicBoolean(false);

    TrackingCloseableSource(int n) {
      docs = new ArrayList<>(n);
      for (int i = 0; i < n; i++) docs.add(IngestDoc.text("id-" + i, "text-" + i));
    }

    boolean iteratorClosed() {
      return closed.get();
    }

    @Override
    public String name() {
      return "tracking";
    }

    @Override
    public java.util.Iterator<IngestDoc> iterator() {
      return new CloseableIterator(docs.iterator(), closed);
    }

    /** Iterator + Closeable (a named type, so it is actually {@code instanceof AutoCloseable}). */
    private static final class CloseableIterator
        implements java.util.Iterator<IngestDoc>, java.io.Closeable {
      private final java.util.Iterator<IngestDoc> inner;
      private final java.util.concurrent.atomic.AtomicBoolean closed;

      CloseableIterator(
          java.util.Iterator<IngestDoc> inner, java.util.concurrent.atomic.AtomicBoolean closed) {
        this.inner = inner;
        this.closed = closed;
      }

      @Override
      public boolean hasNext() {
        return inner.hasNext();
      }

      @Override
      public IngestDoc next() {
        return inner.next();
      }

      @Override
      public void close() {
        closed.set(true);
      }
    }
  }
}
