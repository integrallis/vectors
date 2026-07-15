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
package com.integrallis.vectors.core;

import java.util.Objects;
import java.util.concurrent.Phaser;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/** Persistent workers for the repeated row-parallel operations used during GGUF decoding. */
final class GgufPersistentRowExecutor implements GgufRowExecutor {

  private final int parallelism;
  private final int chunksPerWorker;
  private final Phaser phase;
  private final ReentrantLock publicationLock = new ReentrantLock();
  private final AtomicInteger nextChunk = new AtomicInteger();
  private final AtomicReference<Throwable> failure = new AtomicReference<>();
  private final Thread[] workers;

  private IntConsumer operation;
  private int rows;
  private int chunkSize;
  private boolean closed;

  GgufPersistentRowExecutor(int parallelism, int chunksPerWorker, String threadNamePrefix) {
    if (parallelism < 1) {
      throw new IllegalArgumentException("parallelism must be positive");
    }
    if (chunksPerWorker < 1) {
      throw new IllegalArgumentException("chunksPerWorker must be positive");
    }
    Objects.requireNonNull(threadNamePrefix, "threadNamePrefix");

    this.parallelism = parallelism;
    this.chunksPerWorker = chunksPerWorker;
    this.phase = new Phaser(parallelism);
    this.workers = new Thread[parallelism - 1];
    for (int index = 0; index < workers.length; index++) {
      workers[index] =
          Thread.ofPlatform()
              .daemon()
              .name(threadNamePrefix + '-' + index)
              .unstarted(this::workerLoop);
      workers[index].start();
    }
  }

  @Override
  public void forEach(int rowCount, IntConsumer rowOperation) {
    if (rowCount < 0) {
      throw new IllegalArgumentException("rowCount must not be negative");
    }
    Objects.requireNonNull(rowOperation, "rowOperation");
    if (rowCount == 0) {
      return;
    }

    publicationLock.lock();
    try {
      ensureOpen();
      operation = rowOperation;
      rows = rowCount;
      int targetChunks = (int) Math.min(rowCount, (long) parallelism * chunksPerWorker);
      chunkSize = (rowCount + targetChunks - 1) / targetChunks;
      nextChunk.set(0);
      failure.set(null);

      phase.arriveAndAwaitAdvance();
      executePublishedOperation();
      phase.arriveAndAwaitAdvance();

      Throwable thrown = failure.get();
      operation = null;
      if (thrown != null) {
        rethrow(thrown);
      }
    } finally {
      publicationLock.unlock();
    }
  }

  int parallelism() {
    return parallelism;
  }

  private void workerLoop() {
    while (true) {
      phase.arriveAndAwaitAdvance();
      if (closed) {
        phase.arriveAndAwaitAdvance();
        return;
      }
      executePublishedOperation();
      phase.arriveAndAwaitAdvance();
    }
  }

  private void executePublishedOperation() {
    try {
      while (failure.get() == null) {
        int start = nextChunk.getAndIncrement() * chunkSize;
        if (start >= rows) {
          return;
        }
        int end = Math.min(start + chunkSize, rows);
        for (int row = start; row < end; row++) {
          operation.accept(row);
        }
      }
    } catch (Throwable thrown) {
      failure.compareAndSet(null, thrown);
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("executor is closed");
    }
  }

  private static void rethrow(Throwable thrown) {
    if (thrown instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (thrown instanceof Error error) {
      throw error;
    }
    throw new IllegalStateException("row operation failed", thrown);
  }

  @Override
  public void close() {
    publicationLock.lock();
    try {
      if (closed) {
        return;
      }
      closed = true;
      phase.arriveAndAwaitAdvance();
      phase.arriveAndAwaitAdvance();
    } finally {
      publicationLock.unlock();
    }

    boolean interrupted = false;
    for (Thread worker : workers) {
      while (worker.isAlive()) {
        try {
          worker.join();
        } catch (InterruptedException exception) {
          interrupted = true;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }
}
