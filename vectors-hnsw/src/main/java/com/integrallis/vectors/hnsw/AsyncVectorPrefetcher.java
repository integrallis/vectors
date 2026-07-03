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
package com.integrallis.vectors.hnsw;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Issues asynchronous touch-reads on a {@link RandomAccessVectors} source so that OS page-cache
 * population happens concurrently with graph traversal computation.
 *
 * <h2>Why this works for mmap-backed stores</h2>
 *
 * <p>When {@link RandomAccessVectors} is backed by a {@code MemorySegment} (mmap), each {@code
 * getVector(ordinal)} call that touches a page not yet in the OS page cache triggers a synchronous
 * page fault on the calling thread. Submitting such a read to a background I/O thread causes the OS
 * to load the page before the main search thread needs it, turning a serialised I/O wait into
 * latency that overlaps with distance computation.
 *
 * <h2>Thread safety</h2>
 *
 * <p>{@link RandomAccessVectors#getVector} is called from daemon threads in the pool. The returned
 * {@code float[]} reference is immediately discarded — the sole purpose of the call is to touch the
 * underlying memory and trigger page-in. The {@code MemorySegment} backing the store must be
 * allocated with a shared {@link java.lang.foreign.Arena} (the default for mmap-backed stores in
 * {@code vectors-storage}).
 *
 * <h2>For in-memory stores</h2>
 *
 * <p>When the store is already in RAM (e.g., {@link InMemoryVectors}), the touch-read is a cheap
 * no-op array access. The thread-pool overhead is negligible and correctness is unaffected.
 */
public final class AsyncVectorPrefetcher implements AutoCloseable {

  private final RandomAccessVectors vectors;
  private final ExecutorService ioPool;
  private final AtomicLong submittedCount = new AtomicLong();
  private final AtomicLong failedCount = new AtomicLong();
  private final AtomicReference<Throwable> lastFailure = new AtomicReference<>();
  private final Consumer<Throwable> failureSink;

  /**
   * Creates a prefetcher backed by the given vector source with no failure callback.
   *
   * @param vectors the vector source to prefetch from; must support concurrent reads
   * @param ioThreads number of daemon I/O threads (typically 2–4 for SSD-backed stores)
   */
  public AsyncVectorPrefetcher(RandomAccessVectors vectors, int ioThreads) {
    this(vectors, ioThreads, null);
  }

  /**
   * Creates a prefetcher backed by the given vector source.
   *
   * @param vectors the vector source to prefetch from; must support concurrent reads
   * @param ioThreads number of daemon I/O threads (typically 2–4 for SSD-backed stores)
   * @param failureSink optional callback invoked when a touch-read throws; failures from the sink
   *     itself are caught and discarded so they cannot corrupt the pool thread. {@code null}
   *     disables the callback (failures are still counted and the last cause is retained).
   */
  public AsyncVectorPrefetcher(
      RandomAccessVectors vectors, int ioThreads, Consumer<Throwable> failureSink) {
    this.vectors = vectors;
    this.ioPool = Executors.newFixedThreadPool(ioThreads, new PrefetchThreadFactory());
    this.failureSink = failureSink;
  }

  /**
   * Submits an asynchronous touch-read for the vector at {@code ordinal}.
   *
   * <p>The call returns immediately; the actual memory access happens on a pool thread. Exceptions
   * during the touch (transient SSD/page-fault errors, ordinal out of range that survives the size
   * check, etc.) are caught and recorded — {@link #failedCount()} is incremented, {@link
   * #lastFailure()} is updated, and any configured failure sink is invoked. The main search thread
   * will still see the error if it later touches the same ordinal on the normal path; surfacing the
   * failure here keeps it visible for operators when the search happens to miss the corrupted
   * ordinal.
   *
   * @param ordinal the vector ordinal to prefetch; out-of-range values are silently ignored
   */
  public void prefetch(int ordinal) {
    if (ordinal < 0 || ordinal >= vectors.size()) return;
    submittedCount.incrementAndGet();
    ioPool.submit(
        () -> {
          try {
            // Touch-read: return value discarded; only the memory access matters.
            vectors.getVector(ordinal);
          } catch (Throwable t) {
            failedCount.incrementAndGet();
            lastFailure.set(t);
            Consumer<Throwable> sink = failureSink;
            if (sink != null) {
              try {
                sink.accept(t);
              } catch (Throwable sinkFailure) {
                // A broken sink must not corrupt the pool thread; the failure is already
                // counted via failedCount/lastFailure for observers.
              }
            }
          }
        });
  }

  /**
   * Returns the total number of prefetch requests submitted since construction.
   *
   * <p>Useful for verifying that prefetching is active in tests and benchmarks.
   */
  public long submittedCount() {
    return submittedCount.get();
  }

  /** Returns the number of prefetch touch-reads that threw an exception since construction. */
  public long failedCount() {
    return failedCount.get();
  }

  /**
   * Returns the most recent throwable raised by a prefetch touch-read, or {@code null} if no touch
   * has failed.
   */
  public Throwable lastFailure() {
    return lastFailure.get();
  }

  /**
   * Shuts down the prefetch thread pool. Existing submitted tasks may complete or be cancelled
   * depending on JVM shutdown sequencing.
   *
   * <p>For short-lived search sessions, prefer {@link #close()} via try-with-resources.
   */
  @Override
  public void close() {
    ioPool.shutdownNow();
  }

  // ---------------------------------------------------------------------------
  // Thread factory — daemon threads named for diagnosis
  // ---------------------------------------------------------------------------

  private static final class PrefetchThreadFactory implements ThreadFactory {
    private final AtomicLong count = new AtomicLong();

    @Override
    public Thread newThread(Runnable r) {
      Thread t = new Thread(r, "hnsw-prefetch-" + count.incrementAndGet());
      t.setDaemon(true);
      t.setPriority(Thread.NORM_PRIORITY - 1); // lower than search threads
      return t;
    }
  }
}
