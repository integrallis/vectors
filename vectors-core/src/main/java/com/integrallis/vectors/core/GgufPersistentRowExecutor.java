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
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/** Persistent workers for the repeated row-parallel operations used during GGUF decoding. */
final class GgufPersistentRowExecutor implements GgufRowExecutor {

  private final int parallelism;

  /** Property naming the milliseconds a worker polls at a barrier before it parks (default 5). */
  static final String POLL_MILLIS_PROPERTY = "vectors.gguf.pollMillis";

  /**
   * Nanoseconds a worker polls at a barrier before parking. Read at every barrier so a caller that
   * owns another compute pool can park this executor's workers at run time (see {@link
   * VectorUtil#setGgufPollMillis(long)}); the property sets the initial value.
   */
  private static volatile long pollNanos = configuredPollNanos();

  private final int chunksPerWorker;
  private final Phaser phase;
  private final ReentrantLock publicationLock = new ReentrantLock();
  private final AtomicInteger nextChunk = new AtomicInteger();
  private final AtomicReference<Throwable> failure = new AtomicReference<>();
  private final Thread[] workers;

  private AtomicIntegerArray stageNextChunks = new AtomicIntegerArray(0);
  private IntConsumer operation;
  private GgufStagePlan stagePlan;
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
      stagePlan = null;
      rows = rowCount;
      int targetChunks = (int) Math.min(rowCount, (long) parallelism * chunksPerWorker);
      chunkSize = (rowCount + targetChunks - 1) / targetChunks;
      nextChunk.set(0);
      failure.set(null);

      awaitAdvancePolling();
      executePublishedOperation();
      awaitAdvancePolling();

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

  @Override
  public void execute(GgufStagePlan plan) {
    Objects.requireNonNull(plan, "plan");

    publicationLock.lock();
    try {
      ensureOpen();
      operation = null;
      stagePlan = plan;
      prepareStageChunks(plan.stageCount());
      failure.set(null);

      awaitAdvancePolling();
      executePublishedPlan(plan);

      Throwable thrown = failure.get();
      stagePlan = null;
      if (thrown != null) {
        rethrow(thrown);
      }
    } finally {
      publicationLock.unlock();
    }
  }

  private void workerLoop() {
    while (true) {
      awaitAdvancePolling();
      if (closed) {
        awaitAdvancePolling();
        return;
      }
      GgufStagePlan publishedPlan = stagePlan;
      if (publishedPlan != null) {
        executePublishedPlan(publishedPlan);
      } else {
        executePublishedOperation();
        awaitAdvancePolling();
      }
    }
  }

  private void executePublishedPlan(GgufStagePlan plan) {
    for (int stageIndex = 0; stageIndex < plan.stageCount(); stageIndex++) {
      executeStage(plan.stage(stageIndex), stageIndex);
      awaitAdvancePolling();
    }
  }

  private void executeStage(GgufStagePlan.Stage stage, int stageIndex) {
    if (failure.get() != null) {
      return;
    }
    try {
      int workItems = stage.workItems();
      int targetChunks = (int) Math.min(workItems, (long) parallelism * (long) chunksPerWorker);
      int stageChunkSize = (workItems + targetChunks - 1) / targetChunks;
      while (failure.get() == null) {
        int chunk = stageNextChunks.getAndIncrement(stageIndex);
        if (chunk >= targetChunks) {
          return;
        }
        int start = chunk * stageChunkSize;
        int end = Math.min(start + stageChunkSize, workItems);
        if (start < end) {
          stage.operation().execute(start, end);
        }
      }
    } catch (Throwable thrown) {
      failure.compareAndSet(null, thrown);
    }
  }

  private void prepareStageChunks(int stageCount) {
    if (stageNextChunks.length() < stageCount) {
      stageNextChunks = new AtomicIntegerArray(stageCount);
      return;
    }
    for (int stage = 0; stage < stageCount; stage++) {
      stageNextChunks.set(stage, 0);
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

  /**
   * Arrives at the barrier and waits for the phase to advance, polling for the configured budget
   * before parking. The dispatches of one token are separated by short stretches of single-threaded
   * work; parking the workers across each of them costs a futex wake per worker per dispatch, which
   * measured on a 16-vCPU host as a third of the decode rate. ggml's CPU backend polls 1024 * 128 *
   * 50 relax rounds before a worker sleeps, so its threads never park inside a token; this is the
   * same regime, bounded so an idle executor still parks a few milliseconds after the last
   * dispatch.
   */
  private void awaitAdvancePolling() {
    int arrived = phase.arrive();
    long budget = pollNanos;
    if (budget == 0) {
      phase.awaitAdvance(arrived);
      return;
    }
    long deadline = System.nanoTime() + budget;
    int round = 0;
    while (phase.getPhase() == arrived) {
      if ((++round & 63) == 0) {
        if (System.nanoTime() - deadline >= 0) {
          phase.awaitAdvance(arrived);
          return;
        }
        // On a host with fewer processors than parties, a spinning worker can hold the core the
        // last arriving party needs; yielding at the clock check keeps the barrier from convoying
        // behind its own pollers (the 12-worker close test on a 4-vCPU CI runner timed out
        // without it) and costs one syscall per 64 rounds on an uncontended host.
        Thread.yield();
      }
      Thread.onSpinWait();
    }
  }

  /** Current barrier poll budget in milliseconds (0 parks immediately). */
  static long pollMillis() {
    return pollNanos / 1_000_000L;
  }

  /**
   * Sets the barrier poll budget for every persistent executor in this JVM. 0 parks a worker as
   * soon as it arrives; the upper bound is 60 s. Takes effect at the next barrier.
   */
  static void setPollMillis(long millis) {
    if (millis < 0 || millis > 60_000) {
      throw new IllegalArgumentException(
          "poll budget must be between 0 and 60000 milliseconds: " + millis);
    }
    pollNanos = millis * 1_000_000L;
  }

  static long configuredPollNanos() {
    String configured = System.getProperty(POLL_MILLIS_PROPERTY);
    long millis = 5;
    if (configured != null && !configured.isBlank()) {
      try {
        millis = Long.parseLong(configured.trim());
      } catch (NumberFormatException failure) {
        throw new IllegalArgumentException(
            POLL_MILLIS_PROPERTY + " must be an integer: " + configured, failure);
      }
      if (millis < 0 || millis > 60_000) {
        throw new IllegalArgumentException(
            POLL_MILLIS_PROPERTY + " must be between 0 and 60000 milliseconds: " + configured);
      }
    }
    return millis * 1_000_000L;
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
