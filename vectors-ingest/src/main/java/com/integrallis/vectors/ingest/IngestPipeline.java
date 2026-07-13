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
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Internal producer/consumer pipeline that drives one {@link IngestSource} through embed → batch →
 * commit phases. Owned by {@link BulkIngestor}; one instance is created per {@code ingest(source)}
 * call and disposed when {@link #run()} returns.
 *
 * <p>Threading: a single virtual-thread producer dispatches doc chunks to a virtual-thread embed
 * pool and drains the resulting {@link EmbeddedDoc}s into a bounded queue in source order. The
 * caller of {@link #run()} drains the queue, accumulates batches, and performs the commit (vector
 * sink → sidecart sink → cursor) inline. This keeps the commit path single-threaded — matching the
 * write-lock contract of {@link DistributedVectorCollection}-backed sinks — while still
 * parallelising embedding and source IO.
 */
final class IngestPipeline {

  private static final Logger log = LoggerFactory.getLogger(IngestPipeline.class);

  /** Sentinel pushed onto the queue to signal end-of-stream. Identity-compared by the consumer. */
  private static final EmbeddedDoc EOS_SENTINEL =
      new EmbeddedDoc(IngestDoc.text("__eos__", " "), new float[] {0f}, 0L);

  private final IngestSource source;
  private final Embedder embedder;
  private final VectorSink vectorSink;
  private final SidecartSink sidecartSink;
  private final IngestCursor cursor;
  private final BatchPolicy batchPolicy;
  private final RetryPolicy retryPolicy;
  private final ErrorHandler errorHandler;
  private final int embeddingConcurrency;
  private final int queueCapacity;
  private final int chunkSize;

  private final AtomicLong docsRead = new AtomicLong();
  private final AtomicLong docsEmbedded = new AtomicLong();
  private final AtomicLong docsCommitted = new AtomicLong();
  private final AtomicLong batchesCommitted = new AtomicLong();
  private final AtomicLong bytesEmbedded = new AtomicLong();
  private final AtomicLong retryCount = new AtomicLong();
  private final AtomicReference<String> lastError = new AtomicReference<>();
  private final AtomicReference<Throwable> firstError = new AtomicReference<>();
  private volatile int currentQueueDepth;

  IngestPipeline(
      IngestSource source,
      Embedder embedder,
      VectorSink vectorSink,
      SidecartSink sidecartSink,
      IngestCursor cursor,
      BatchPolicy batchPolicy,
      RetryPolicy retryPolicy,
      ErrorHandler errorHandler,
      int embeddingConcurrency,
      int queueCapacity) {
    this.source = Objects.requireNonNull(source, "source");
    this.embedder = Objects.requireNonNull(embedder, "embedder");
    this.vectorSink = Objects.requireNonNull(vectorSink, "vectorSink");
    this.sidecartSink = Objects.requireNonNull(sidecartSink, "sidecartSink");
    this.cursor = Objects.requireNonNull(cursor, "cursor");
    this.batchPolicy = Objects.requireNonNull(batchPolicy, "batchPolicy");
    this.retryPolicy = Objects.requireNonNull(retryPolicy, "retryPolicy");
    this.errorHandler = Objects.requireNonNull(errorHandler, "errorHandler");
    if (embeddingConcurrency <= 0) {
      throw new IllegalArgumentException("embeddingConcurrency must be > 0");
    }
    if (queueCapacity <= 0) {
      throw new IllegalArgumentException("queueCapacity must be > 0");
    }
    this.embeddingConcurrency = embeddingConcurrency;
    this.queueCapacity = queueCapacity;
    this.chunkSize = Math.max(1, Math.min(64, batchPolicy.maxDocs() / 4));
  }

  IngestResult run() throws IOException, InterruptedException {
    long startNs = System.nanoTime();
    LinkedBlockingQueue<EmbeddedDoc> queue = new LinkedBlockingQueue<>(queueCapacity);
    AtomicReference<Throwable> producerError = new AtomicReference<>();
    BatchAccumulator accumulator = new BatchAccumulator(batchPolicy);
    long sourceStart = source.startOffset();
    // Durable resume: seed from the persisted cursor so a restart does not re-ingest from the
    // start.
    // The cursor holds the last committed 0-based offset (0 is also the "no cursor" default), so we
    // resume at cursor+1 when there is real progress, never earlier than the source's own baseline.
    long persisted = cursor.load(source.name());
    long startOffset = Math.max(sourceStart, persisted > 0 ? persisted + 1 : 0);
    // Docs already committed in a prior run beyond the source's own skip — advanced past in the
    // producer so they are not re-embedded/re-committed.
    long resumeSkip = startOffset - sourceStart;
    long lastSavedOffset = persisted > 0 ? persisted : sourceStart;

    try (ExecutorService embedExec = Executors.newVirtualThreadPerTaskExecutor()) {
      Thread producer =
          Thread.ofVirtual()
              .name("ingest-producer-" + source.name())
              .start(() -> runProducer(queue, embedExec, producerError, startOffset, resumeSkip));
      try {
        long pollTimeoutNs = Math.max(1_000_000L, batchPolicy.maxLatency().toNanos() / 4);
        while (true) {
          EmbeddedDoc d = queue.poll(pollTimeoutNs, TimeUnit.NANOSECONDS);
          currentQueueDepth = queue.size();
          if (d == EOS_SENTINEL) break;
          if (d != null) {
            Optional<Batch> ready = accumulator.add(d);
            if (ready.isPresent()) lastSavedOffset = commitBatch(ready.get());
          } else {
            Optional<Batch> ready = accumulator.flushIfTimedOut(System.nanoTime());
            if (ready.isPresent()) lastSavedOffset = commitBatch(ready.get());
          }
        }
        Optional<Batch> last = accumulator.drain();
        if (last.isPresent()) lastSavedOffset = commitBatch(last.get());
      } finally {
        producer.join();
      }
      Throwable pe = producerError.get();
      if (pe != null) firstError.compareAndSet(null, pe);
    }

    Duration total = Duration.ofNanos(System.nanoTime() - startNs);
    return new IngestResult(
        docsRead.get(),
        docsEmbedded.get(),
        docsCommitted.get(),
        batchesCommitted.get(),
        bytesEmbedded.get(),
        total,
        lastSavedOffset,
        Optional.ofNullable(firstError.get()));
  }

  /** Live, point-in-time metrics snapshot. Safe to call from any thread. */
  IngestMetrics snapshot() {
    return new IngestMetrics(
        docsRead.get(),
        docsEmbedded.get(),
        docsCommitted.get(),
        batchesCommitted.get(),
        bytesEmbedded.get(),
        currentQueueDepth,
        queueCapacity,
        retryCount.get(),
        Optional.ofNullable(lastError.get()));
  }

  // ─── producer ─────────────────────────────────────────────────────────────

  private void runProducer(
      LinkedBlockingQueue<EmbeddedDoc> queue,
      ExecutorService embedExec,
      AtomicReference<Throwable> errorSlot,
      long startOffset,
      long resumeSkip) {
    Iterator<IngestDoc> it = null;
    try {
      it = source.iterator();
      // The source iterator has already skipped source.startOffset(); advance past any additional
      // already-committed docs (durable resume) so they are not re-ingested. These are not counted
      // in docsRead — they were read and committed on the prior run.
      for (long i = 0; i < resumeSkip && it.hasNext(); i++) {
        it.next();
      }
      long offset = startOffset;
      Deque<Future<List<EmbeddedDoc>>> pending = new ArrayDeque<>();
      int maxInFlight = Math.max(1, embeddingConcurrency * 2);

      while (it.hasNext()) {
        List<IngestDoc> chunk = new ArrayList<>(chunkSize);
        List<Long> chunkOffsets = new ArrayList<>(chunkSize);
        while (chunk.size() < chunkSize && it.hasNext()) {
          IngestDoc d = it.next();
          docsRead.incrementAndGet();
          chunk.add(d);
          chunkOffsets.add(offset++);
        }
        final List<IngestDoc> finalChunk = List.copyOf(chunk);
        final List<Long> finalOffsets = List.copyOf(chunkOffsets);
        pending.offer(embedExec.submit(() -> embedChunk(finalChunk, finalOffsets)));
        while (pending.size() >= maxInFlight) {
          drainHeadInto(pending, queue);
        }
      }
      while (!pending.isEmpty()) {
        drainHeadInto(pending, queue);
      }
    } catch (Throwable t) {
      errorSlot.compareAndSet(null, t);
      lastError.compareAndSet(null, t.getMessage() != null ? t.getMessage() : t.toString());
    } finally {
      // Release the source iterator's resources (e.g. a JsonlSource file descriptor) whether we
      // drained it to EOF or aborted early on error/backpressure. Without this, a source whose
      // iterator only self-closes on natural EOF leaks one FD per aborted ingest.
      if (it instanceof AutoCloseable ac) {
        try {
          ac.close();
        } catch (Exception closeError) {
          log.warn("Failed to close source iterator for '{}'", source.name(), closeError);
        }
      }
      try {
        queue.put(EOS_SENTINEL);
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private List<EmbeddedDoc> embedChunk(List<IngestDoc> chunk, List<Long> offsets) {
    List<IngestDoc> toEmbed = new ArrayList<>(chunk.size());
    for (IngestDoc d : chunk) {
      if (d.precomputedVector() == null) toEmbed.add(d);
    }
    List<float[]> embedded = toEmbed.isEmpty() ? List.of() : embedder.embedAll(toEmbed);
    List<EmbeddedDoc> out = new ArrayList<>(chunk.size());
    int embIdx = 0;
    for (int i = 0; i < chunk.size(); i++) {
      IngestDoc d = chunk.get(i);
      float[] v = d.precomputedVector();
      if (v == null) v = embedded.get(embIdx++);
      out.add(new EmbeddedDoc(d, v, offsets.get(i)));
      docsEmbedded.incrementAndGet();
      bytesEmbedded.addAndGet(BatchAccumulator.sizeOf(d));
    }
    return out;
  }

  private void drainHeadInto(
      Deque<Future<List<EmbeddedDoc>>> pending, LinkedBlockingQueue<EmbeddedDoc> queue)
      throws InterruptedException {
    Future<List<EmbeddedDoc>> head = pending.poll();
    if (head == null) return;
    List<EmbeddedDoc> docs;
    try {
      docs = head.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause() != null ? e.getCause() : e;
      if (cause instanceof RuntimeException re) throw re;
      if (cause instanceof Error err) throw err;
      throw new RuntimeException(cause);
    }
    for (EmbeddedDoc d : docs) {
      queue.put(d);
      currentQueueDepth = queue.size();
    }
  }

  // ─── commit ───────────────────────────────────────────────────────────────

  private long commitBatch(Batch batch) throws IOException {
    try {
      retryPolicy.execute(
          () -> {
            vectorSink.addAll(batch);
            return null;
          });
      retryPolicy.execute(
          () -> {
            vectorSink.commit();
            return null;
          });
      retryPolicy.execute(
          () -> {
            sidecartSink.writeAll(batch);
            return null;
          });
      cursor.save(source.name(), batch.lastSourceOffset());
    } catch (Exception e) {
      lastError.compareAndSet(null, e.getMessage() != null ? e.getMessage() : e.toString());
      ErrorHandler.IngestErrorContext ctx =
          new ErrorHandler.IngestErrorContext("commit", batch.batchId(), e);
      ErrorHandler.Decision decision = errorHandler.onError(ctx);
      firstError.compareAndSet(null, e);
      if (decision == ErrorHandler.Decision.FAIL_FAST) {
        if (e instanceof IOException io) throw io;
        if (e instanceof RuntimeException re) throw re;
        throw new IOException("commit failed for batch " + batch.batchId(), e);
      }
      log.warn("ingest: continuing past commit failure on batch {}", batch.batchId(), e);
      return batch.lastSourceOffset();
    }
    docsCommitted.addAndGet(batch.size());
    batchesCommitted.incrementAndGet();
    return batch.lastSourceOffset();
  }
}
