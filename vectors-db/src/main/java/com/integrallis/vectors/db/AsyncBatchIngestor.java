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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Asynchronous batch ingestion into a {@link VectorCollection} (I.8). Documents are handed off to a
 * background worker that batches {@code add}s and {@code commit}s off the producer thread, so a
 * producer never blocks on the (serial, fsync-bound) commit path — only on the bounded internal
 * buffer when it outruns the worker (natural backpressure). The ingest completes with a {@link
 * CommitToken} via {@link #completion()}.
 *
 * <p>Two ways to feed it, both safe to mix:
 *
 * <ul>
 *   <li><b>Direct:</b> {@link #submit(Document)} per document, then {@link #close()} to flush the
 *       final partial batch and finish.
 *   <li><b>Reactive:</b> register as a {@link Flow.Subscriber} on a {@code
 *       Flow.Publisher<Document>} (e.g. bridging a {@code vectors-ingest} source); demand is
 *       requested in batch-sized chunks.
 * </ul>
 *
 * <p>Commits are ordered (single worker). A failed {@code add}/{@code commit}, an {@code onError}
 * from the publisher, completes {@link #completion()} exceptionally and stops the worker;
 * subsequent {@link #submit} calls throw. The worker thread is a daemon.
 */
public final class AsyncBatchIngestor implements Flow.Subscriber<Document>, AutoCloseable {

  private sealed interface Item {}

  private record DocItem(Document doc) implements Item {}

  private record CompleteItem() implements Item {}

  private record ErrorItem(Throwable error) implements Item {}

  private final VectorCollection collection;
  private final int batchSize;
  private final boolean upsert;
  private final BlockingQueue<Item> queue;
  private final Thread worker;
  private final CompletableFuture<CommitToken> completion = new CompletableFuture<>();
  private final AtomicBoolean terminated = new AtomicBoolean(false);
  private volatile Flow.Subscription subscription;

  /** Ingests via {@link VectorCollection#add} (rejects duplicate ids), committing every batch. */
  public AsyncBatchIngestor(VectorCollection collection, int batchSize) {
    this(collection, batchSize, false);
  }

  /**
   * @param collection the target collection
   * @param batchSize documents per commit; must be {@code >= 1}
   * @param upsert when true use {@link VectorCollection#upsert} (insert-or-replace) instead of
   *     {@link VectorCollection#add}
   */
  public AsyncBatchIngestor(VectorCollection collection, int batchSize, boolean upsert) {
    this.collection = Objects.requireNonNull(collection, "collection must not be null");
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    this.batchSize = batchSize;
    this.upsert = upsert;
    this.queue = new LinkedBlockingQueue<>(Math.max(2 * batchSize, batchSize + 1));
    this.worker = new Thread(this::runWorker, "vectors-async-ingest");
    this.worker.setDaemon(true);
    this.worker.start();
  }

  /**
   * Hands a document to the worker, blocking only while the internal buffer is full (backpressure).
   *
   * @throws IllegalStateException if the ingest has already completed or failed
   */
  public void submit(Document doc) {
    Objects.requireNonNull(doc, "doc must not be null");
    if (terminated.get()) {
      throw new IllegalStateException("ingestor already completed; no more documents accepted");
    }
    put(new DocItem(doc));
  }

  /**
   * The eventual ingest result; completes after the final commit, or exceptionally on failure.
   * Returned as a read-only {@link CompletionStage} (it cannot complete the underlying ingest).
   */
  public CompletionStage<CommitToken> completion() {
    return completion.minimalCompletionStage();
  }

  // --- Flow.Subscriber ---

  @Override
  public void onSubscribe(Flow.Subscription sub) {
    this.subscription = Objects.requireNonNull(sub, "subscription must not be null");
    sub.request(batchSize); // bounded initial demand
  }

  @Override
  public void onNext(Document doc) {
    submit(doc);
    Flow.Subscription sub = this.subscription;
    if (sub != null) {
      sub.request(1); // one-for-one replenishment keeps demand ~= batchSize
    }
  }

  @Override
  public void onError(Throwable throwable) {
    if (terminated.compareAndSet(false, true)) {
      offerTerminal(new ErrorItem(throwable));
    }
  }

  @Override
  public void onComplete() {
    if (terminated.compareAndSet(false, true)) {
      offerTerminal(new CompleteItem());
    }
  }

  /**
   * Signals completion (flush the final partial batch and commit), then waits for the worker to
   * finish. Idempotent; safe to call after {@link #onComplete}/{@link #onError}.
   */
  @Override
  public void close() {
    if (terminated.compareAndSet(false, true)) {
      offerTerminal(new CompleteItem());
    }
    try {
      worker.join();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  // --- internals ---

  private void put(Item item) {
    try {
      while (!completion.isDone()) {
        if (queue.offer(item, 50, TimeUnit.MILLISECONDS)) {
          return;
        }
      }
      // The worker has stopped (completed or failed) — don't block forever.
      throw new IllegalStateException("async ingestor has terminated");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("interrupted while enqueuing document", e);
    }
  }

  private void offerTerminal(Item item) {
    if (completion.isDone()) {
      return; // worker already finished (e.g. failed mid-stream); nothing to signal
    }
    try {
      put(item);
    } catch (IllegalStateException ignored) {
      // Worker terminated concurrently; completion is already settled.
    }
  }

  private void runWorker() {
    long committed = 0L;
    int pending = 0;
    try {
      while (true) {
        Item item = queue.take();
        if (item instanceof DocItem d) {
          if (upsert) {
            collection.upsert(d.doc());
          } else {
            collection.add(d.doc());
          }
          if (++pending >= batchSize) {
            collection.commit();
            committed += pending;
            pending = 0;
          }
        } else if (item instanceof CompleteItem) {
          if (pending > 0) {
            collection.commit();
            committed += pending;
          }
          completion.complete(new CommitToken(collection.generationNumber(), committed));
          return;
        } else if (item instanceof ErrorItem e) {
          completion.completeExceptionally(e.error());
          return;
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      completion.completeExceptionally(e);
    } catch (RuntimeException e) {
      completion.completeExceptionally(e);
    }
  }
}
