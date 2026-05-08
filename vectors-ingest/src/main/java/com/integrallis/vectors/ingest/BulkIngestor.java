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

import com.integrallis.vectors.ingest.cursor.InMemoryCursor;
import com.integrallis.vectors.ingest.sinks.NoopSidecartSink;
import java.io.IOException;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Public façade that wires an {@link IngestSource} through {@link Embedder} → batched commit to a
 * {@link VectorSink} (and optional {@link SidecartSink}), persisting resume state through an {@link
 * IngestCursor}. Build via {@link #builder()}, drive with {@link #ingest(IngestSource)}, and close
 * to release the underlying sinks.
 *
 * <p>A single {@code BulkIngestor} instance can be used for many sequential {@link
 * #ingest(IngestSource) ingest} calls; each call runs synchronously and drains all in-flight
 * batches before returning. Concurrent calls into the same instance are not supported.
 */
public final class BulkIngestor implements AutoCloseable {

  private final VectorSink vectorSink;
  private final SidecartSink sidecartSink;
  private final Embedder embedder;
  private final BatchPolicy batchPolicy;
  private final int embeddingConcurrency;
  private final int sinkConcurrency;
  private final int queueCapacity;
  private final IngestCursor cursor;
  private final RetryPolicy retryPolicy;
  private final ErrorHandler errorHandler;
  private final AtomicReference<IngestPipeline> active = new AtomicReference<>();
  private volatile boolean closed;

  private BulkIngestor(Builder b) {
    this.vectorSink = b.vectorSink;
    this.sidecartSink = b.sidecartSink;
    this.embedder = b.embedder;
    this.batchPolicy = b.batchPolicy;
    this.embeddingConcurrency = b.embeddingConcurrency;
    this.sinkConcurrency = b.sinkConcurrency;
    this.queueCapacity = b.queueCapacity;
    this.cursor = b.cursor;
    this.retryPolicy = b.retryPolicy;
    this.errorHandler = b.errorHandler;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Drives {@code source} through embed → batch → commit until exhausted. Returns aggregate
   * counters; on fail-fast errors throws an {@link IOException} or wraps an underlying {@link
   * RuntimeException}.
   */
  public IngestResult ingest(IngestSource source) throws IOException {
    Objects.requireNonNull(source, "source");
    if (closed) {
      throw new IllegalStateException("BulkIngestor is closed");
    }
    IngestPipeline pipeline =
        new IngestPipeline(
            source,
            embedder,
            vectorSink,
            sidecartSink,
            cursor,
            batchPolicy,
            retryPolicy,
            errorHandler,
            embeddingConcurrency,
            queueCapacity);
    active.set(pipeline);
    try {
      return pipeline.run();
    } catch (InterruptedException ie) {
      Thread.currentThread().interrupt();
      throw new IOException("ingestion interrupted", ie);
    }
  }

  /** Live snapshot from the in-flight (or most recently completed) ingest. */
  public IngestMetrics metrics() {
    IngestPipeline p = active.get();
    if (p == null) {
      return new IngestMetrics(0, 0, 0, 0, 0, 0, queueCapacity, 0, Optional.empty());
    }
    return p.snapshot();
  }

  @Override
  public void close() throws IOException {
    if (closed) return;
    closed = true;
    IOException combined = null;
    try {
      vectorSink.close();
    } catch (IOException e) {
      combined = e;
    }
    try {
      sidecartSink.close();
    } catch (IOException e) {
      if (combined != null) combined.addSuppressed(e);
      else combined = e;
    }
    if (combined != null) throw combined;
  }

  // ─── builder ─────────────────────────────────────────────────────────────

  public static final class Builder {
    private VectorSink vectorSink;
    private SidecartSink sidecartSink = new NoopSidecartSink();
    private Embedder embedder;
    private BatchPolicy batchPolicy = BatchPolicy.defaults();
    private int embeddingConcurrency = Math.max(1, Runtime.getRuntime().availableProcessors());
    private int sinkConcurrency = 2;
    private int queueCapacity = -1;
    private IngestCursor cursor = new InMemoryCursor();
    private RetryPolicy retryPolicy = RetryPolicy.defaults();
    private ErrorHandler errorHandler = ErrorHandler.failFast();

    public Builder vectorSink(VectorSink sink) {
      this.vectorSink = Objects.requireNonNull(sink, "vectorSink");
      return this;
    }

    public Builder sidecartSink(SidecartSink sink) {
      this.sidecartSink = Objects.requireNonNull(sink, "sidecartSink");
      return this;
    }

    public Builder embedder(Embedder e) {
      this.embedder = Objects.requireNonNull(e, "embedder");
      return this;
    }

    public Builder batchPolicy(BatchPolicy p) {
      this.batchPolicy = Objects.requireNonNull(p, "batchPolicy");
      return this;
    }

    public Builder embeddingConcurrency(int n) {
      if (n <= 0) throw new IllegalArgumentException("embeddingConcurrency must be > 0");
      this.embeddingConcurrency = n;
      return this;
    }

    public Builder sinkConcurrency(int n) {
      if (n <= 0) throw new IllegalArgumentException("sinkConcurrency must be > 0");
      this.sinkConcurrency = n;
      return this;
    }

    public Builder queueCapacity(int n) {
      if (n <= 0) throw new IllegalArgumentException("queueCapacity must be > 0");
      this.queueCapacity = n;
      return this;
    }

    public Builder cursor(IngestCursor c) {
      this.cursor = Objects.requireNonNull(c, "cursor");
      return this;
    }

    public Builder retryPolicy(RetryPolicy p) {
      this.retryPolicy = Objects.requireNonNull(p, "retryPolicy");
      return this;
    }

    public Builder onError(ErrorHandler h) {
      this.errorHandler = Objects.requireNonNull(h, "errorHandler");
      return this;
    }

    public BulkIngestor build() {
      if (vectorSink == null) {
        throw new IllegalStateException("vectorSink is required");
      }
      if (embedder == null) {
        throw new IllegalStateException(
            "embedder is required (use PrecomputedEmbedder if all docs ship vectors)");
      }
      if (queueCapacity <= 0) {
        queueCapacity = Math.max(8, embeddingConcurrency * 4);
      }
      return new BulkIngestor(this);
    }
  }
}
