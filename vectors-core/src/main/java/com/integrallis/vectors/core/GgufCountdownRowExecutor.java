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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/**
 * Persistent GGUF workers that publish a job by bumping a generation counter instead of meeting at
 * a {@link java.util.concurrent.Phaser}.
 *
 * <p>Why this exists alongside {@link GgufPersistentRowExecutor}: a Phaser keeps the arrival count
 * and the release signal in <i>one</i> word. Every party CASes that word in {@code arrive()} while
 * every waiting party spin-reads it in {@code getPhase()}, so each of the N-1 pollers invalidates
 * the cache line the arrivers need to own, and a dispatch pays that twice because a published
 * operation needs a publication barrier and a completion barrier. The cost grows with thread count,
 * which is the wrong direction for the thing it is meant to accelerate.
 *
 * <p>Here the two roles live on separate cache lines and nothing is both CASed and polled by many
 * threads at once:
 *
 * <ul>
 *   <li>Workers poll {@code GENERATION}, which the publishing caller writes exactly once per
 *       dispatch. No CAS ever touches it, so N-1 pollers read a line that stays in shared state.
 *   <li>Completion is a countdown on {@code REMAINING}, which each participant decrements once.
 *       Only the caller polls it, so the N decrements contend with a single reader rather than with
 *       N-1.
 * </ul>
 *
 * <p>That also removes the second barrier outright. A worker that finishes early does not have to
 * wait for its peers: it loops back, sees the generation it already ran, and waits for the next
 * one. Ordered stage plans still need a meeting point between stages, so those use a counting
 * barrier built the same way -- arrivals decrement one line, the release is a monotonic counter on
 * another.
 *
 * <p>Chunking, the poll-before-park budget and caller participation are deliberately identical to
 * {@link GgufPersistentRowExecutor} so that comparing the two measures the publication mechanism
 * and not a different work split. Selected with {@code -Dvectors.gguf.executor=countdown}.
 */
final class GgufCountdownRowExecutor implements GgufRowExecutor {

  private static final VarHandle COUNTERS = MethodHandles.arrayElementVarHandle(long[].class);

  /**
   * Counter slots, spaced eight longs apart so no two share a 64-byte cache line, with slot 0 left
   * as a leading pad against whatever the allocator puts in front of the array.
   */
  private static final int GENERATION = 8;

  private static final int REMAINING = 16;
  private static final int NEXT_CHUNK = 24;
  private static final int PARKED = 32;
  private static final int STAGE_RELEASE = 40;
  private static final int STAGE_REMAINING = 48;
  private static final int COUNTER_SLOTS = 56;

  private final int parallelism;
  private final int chunksPerWorker;
  private final long[] counters = new long[COUNTER_SLOTS + 8];
  private final ReentrantLock publicationLock = new ReentrantLock();
  private final ReentrantLock parkLock = new ReentrantLock();
  private final Condition parkSignal = parkLock.newCondition();
  private final AtomicReference<Throwable> failure = new AtomicReference<>();
  private final Worker[] workers;

  private IntConsumer operation;
  private GgufStagePlan stagePlan;
  private int rows;
  private int chunkSize;
  private volatile boolean closed;

  GgufCountdownRowExecutor(int parallelism, int chunksPerWorker, String threadNamePrefix) {
    if (parallelism < 1) {
      throw new IllegalArgumentException("parallelism must be positive");
    }
    if (chunksPerWorker < 1) {
      throw new IllegalArgumentException("chunksPerWorker must be positive");
    }
    Objects.requireNonNull(threadNamePrefix, "threadNamePrefix");
    this.parallelism = parallelism;
    this.chunksPerWorker = chunksPerWorker;
    this.workers = new Worker[parallelism - 1];
    for (int index = 0; index < workers.length; index++) {
      workers[index] = new Worker(this, threadNamePrefix + '-' + index);
      workers[index].start();
    }
  }

  /**
   * A pool thread that carries the nesting flag as a field, so a worker neither probes nor writes a
   * {@code ThreadLocalMap} on either side of every chunk run.
   */
  private static final class Worker extends Thread implements GgufCallbackScope.Flagged {

    private final GgufCountdownRowExecutor owner;

    /**
     * Read and written only by this thread, so it needs no memory ordering.
     *
     * <p>{@link GgufCallbackScope} always resolves the flag through {@code Thread.currentThread()},
     * so the only thread that can see this field is the one it belongs to -- that is the whole
     * reason it can be a plain field instead of a thread local. Making it volatile would buy
     * nothing and would suggest cross-thread access is intended.
     */
    private boolean inCallback;

    Worker(GgufCountdownRowExecutor owner, String name) {
      super(name);
      this.owner = owner;
      setDaemon(true);
    }

    @Override
    public boolean inGgufCallback() {
      return inCallback;
    }

    @Override
    public void inGgufCallback(boolean value) {
      inCallback = value;
    }

    @Override
    public void run() {
      owner.workerLoop();
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
    if (GgufCallbackScope.active()) {
      ensureOpen();
      for (int row = 0; row < rowCount; row++) {
        rowOperation.accept(row);
      }
      return;
    }

    publicationLock.lock();
    try {
      ensureOpen();
      operation = rowOperation;
      stagePlan = null;
      rows = rowCount;
      int targetChunks = (int) Math.min(rowCount, (long) parallelism * chunksPerWorker);
      chunkSize = 1 + (rowCount - 1) / targetChunks;
      publish();
      runPublishedOperation();
      awaitCompletion();
      operation = null;
      Throwable thrown = failure.get();
      if (thrown != null) {
        rethrow(thrown);
      }
    } finally {
      publicationLock.unlock();
    }
  }

  @Override
  public void execute(GgufStagePlan plan) {
    Objects.requireNonNull(plan, "plan");
    if (GgufCallbackScope.active()) {
      ensureOpen();
      plan.executeSerially();
      return;
    }

    publicationLock.lock();
    try {
      ensureOpen();
      operation = null;
      stagePlan = plan;
      publish();
      runPublishedPlan(plan);
      awaitCompletion();
      stagePlan = null;
      Throwable thrown = failure.get();
      if (thrown != null) {
        rethrow(thrown);
      }
    } finally {
      publicationLock.unlock();
    }
  }

  /**
   * Makes the published job visible. Every field the workers read is written before the generation
   * bump, and the bump is the release that pairs with each worker's acquiring poll.
   */
  private void publish() {
    failure.set(null);
    COUNTERS.setVolatile(counters, NEXT_CHUNK, 0L);
    COUNTERS.setVolatile(counters, STAGE_REMAINING, (long) parallelism);
    COUNTERS.setVolatile(counters, STAGE_RELEASE, 0L);
    COUNTERS.setVolatile(counters, REMAINING, (long) parallelism);
    COUNTERS.getAndAdd(counters, GENERATION, 1L);
    wakeParked();
  }

  private void workerLoop() {
    long seen = 0L;
    while (true) {
      seen = awaitGeneration(seen);
      if (closed) {
        return;
      }
      GgufStagePlan published = stagePlan;
      if (published != null) {
        runPublishedPlan(published);
      } else {
        runPublishedOperation();
      }
    }
  }

  /** Polls for a generation other than {@code seen}, parking once the poll budget runs out. */
  private long awaitGeneration(long seen) {
    long current = (long) COUNTERS.getVolatile(counters, GENERATION);
    if (current != seen) {
      return current;
    }
    long budget = GgufPersistentRowExecutor.pollNanos();
    long deadline = System.nanoTime() + budget;
    int round = 0;
    while (true) {
      current = (long) COUNTERS.getVolatile(counters, GENERATION);
      if (current != seen) {
        return current;
      }
      if (budget == 0 || ((++round & 63) == 0 && System.nanoTime() - deadline >= 0)) {
        return parkUntilGeneration(seen);
      }
      if ((round & 63) == 0) {
        // A host with fewer processors than parties can have a poller holding the core the
        // publisher needs; yielding at the clock check keeps a dispatch from convoying behind
        // its own pollers.
        Thread.yield();
      }
      Thread.onSpinWait();
    }
  }

  private long parkUntilGeneration(long seen) {
    parkLock.lock();
    try {
      COUNTERS.getAndAdd(counters, PARKED, 1L);
      try {
        long current = (long) COUNTERS.getVolatile(counters, GENERATION);
        while (current == seen) {
          parkSignal.awaitUninterruptibly();
          current = (long) COUNTERS.getVolatile(counters, GENERATION);
        }
        return current;
      } finally {
        COUNTERS.getAndAdd(counters, PARKED, -1L);
      }
    } finally {
      parkLock.unlock();
    }
  }

  /** Waits for every participant to finish the published job. */
  private void awaitCompletion() {
    if ((long) COUNTERS.getVolatile(counters, REMAINING) == 0L) {
      return;
    }
    long budget = GgufPersistentRowExecutor.pollNanos();
    long deadline = System.nanoTime() + budget;
    int round = 0;
    while ((long) COUNTERS.getVolatile(counters, REMAINING) != 0L) {
      if (budget == 0 || ((++round & 63) == 0 && System.nanoTime() - deadline >= 0)) {
        parkUntilComplete();
        return;
      }
      if ((round & 63) == 0) {
        Thread.yield();
      }
      Thread.onSpinWait();
    }
  }

  private void parkUntilComplete() {
    parkLock.lock();
    try {
      COUNTERS.getAndAdd(counters, PARKED, 1L);
      try {
        while ((long) COUNTERS.getVolatile(counters, REMAINING) != 0L) {
          parkSignal.awaitUninterruptibly();
        }
      } finally {
        COUNTERS.getAndAdd(counters, PARKED, -1L);
      }
    } finally {
      parkLock.unlock();
    }
  }

  /** Signals parked threads, taking the lock only when one is actually parked. */
  private void wakeParked() {
    if ((long) COUNTERS.getVolatile(counters, PARKED) == 0L) {
      return;
    }
    parkLock.lock();
    try {
      parkSignal.signalAll();
    } finally {
      parkLock.unlock();
    }
  }

  /** Records this participant as finished, waking the caller when it was the last one. */
  private void finishParticipation() {
    long left = (long) COUNTERS.getAndAdd(counters, REMAINING, -1L) - 1L;
    if (left == 0L) {
      wakeParked();
    }
  }

  private void runPublishedOperation() {
    GgufCallbackScope.enter();
    try {
      IntConsumer published = operation;
      int rowCount = rows;
      int size = chunkSize;
      while (failure.get() == null) {
        long start = (long) COUNTERS.getAndAdd(counters, NEXT_CHUNK, 1L) * size;
        if (start < 0L || start >= rowCount) {
          break;
        }
        int end = (int) Math.min(start + size, rowCount);
        for (int row = (int) start; row < end; row++) {
          published.accept(row);
        }
      }
    } catch (Throwable thrown) {
      failure.compareAndSet(null, thrown);
    } finally {
      GgufCallbackScope.leave();
      finishParticipation();
    }
  }

  private void runPublishedPlan(GgufStagePlan plan) {
    GgufCallbackScope.enter();
    try {
      int stageCount = plan.stageCount();
      for (int stageIndex = 0; stageIndex < stageCount; stageIndex++) {
        runStage(plan.stage(stageIndex));
        if (stageIndex + 1 < stageCount) {
          stageBarrier();
        }
      }
    } finally {
      GgufCallbackScope.leave();
      finishParticipation();
    }
  }

  private void runStage(GgufStagePlan.Stage stage) {
    if (failure.get() != null) {
      return;
    }
    try {
      int workItems = stage.workItems();
      int targetChunks = (int) Math.min(workItems, (long) parallelism * (long) chunksPerWorker);
      int stageChunkSize = 1 + (workItems - 1) / targetChunks;
      while (failure.get() == null) {
        long start = (long) COUNTERS.getAndAdd(counters, NEXT_CHUNK, 1L) * stageChunkSize;
        if (start < 0L || start >= workItems) {
          return;
        }
        int end = (int) Math.min(start + stageChunkSize, workItems);
        stage.operation().execute((int) start, end);
      }
    } catch (Throwable thrown) {
      failure.compareAndSet(null, thrown);
    }
  }

  /**
   * A counting barrier between ordered stages, split across two cache lines for the same reason the
   * publication is: arrivals decrement {@code STAGE_REMAINING}, and the last arriver resets it and
   * bumps {@code STAGE_RELEASE}, which is the only line the waiters poll. Resetting the count
   * before the release bump is what makes the next stage's barrier safe to enter immediately.
   */
  private void stageBarrier() {
    long release = (long) COUNTERS.getVolatile(counters, STAGE_RELEASE);
    long left = (long) COUNTERS.getAndAdd(counters, STAGE_REMAINING, -1L) - 1L;
    if (left == 0L) {
      COUNTERS.setVolatile(counters, NEXT_CHUNK, 0L);
      COUNTERS.setVolatile(counters, STAGE_REMAINING, (long) parallelism);
      COUNTERS.setVolatile(counters, STAGE_RELEASE, release + 1L);
      wakeParked();
      return;
    }
    long budget = GgufPersistentRowExecutor.pollNanos();
    long deadline = System.nanoTime() + budget;
    int round = 0;
    while ((long) COUNTERS.getVolatile(counters, STAGE_RELEASE) == release) {
      if (budget == 0 || ((++round & 63) == 0 && System.nanoTime() - deadline >= 0)) {
        parkUntilStageRelease(release);
        return;
      }
      if ((round & 63) == 0) {
        Thread.yield();
      }
      Thread.onSpinWait();
    }
  }

  private void parkUntilStageRelease(long release) {
    parkLock.lock();
    try {
      COUNTERS.getAndAdd(counters, PARKED, 1L);
      try {
        while ((long) COUNTERS.getVolatile(counters, STAGE_RELEASE) == release) {
          parkSignal.awaitUninterruptibly();
        }
      } finally {
        COUNTERS.getAndAdd(counters, PARKED, -1L);
      }
    } finally {
      parkLock.unlock();
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
    // A callback cannot wait for a publication that includes itself, and cross-pool callbacks can
    // form the same cycle, so closing must happen outside all row callbacks.
    if (GgufCallbackScope.active()) {
      throw new IllegalStateException("cannot close an executor from a row callback");
    }
    publicationLock.lock();
    try {
      if (closed) {
        return;
      }
      closed = true;
      // Workers check `closed` the moment they observe a new generation, so one bump releases
      // every one of them whether it was polling or parked.
      COUNTERS.getAndAdd(counters, GENERATION, 1L);
      wakeParked();
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
