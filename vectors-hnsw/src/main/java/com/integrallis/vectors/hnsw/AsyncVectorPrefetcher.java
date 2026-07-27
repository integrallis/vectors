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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
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
  // Zero-copy touch path: when the store exposes stable mmap slices, page-in by reading one byte
  // per
  // 4 KiB page off the slice instead of calling getVector() — which on a build-safe store allocates
  // a fresh float[dim] per touch and would reintroduce GC churn on the I/O pool at 100M scale.
  private final boolean useSegments;
  private final int rawVectorBytes;
  private static final long PAGE = 4096;
  // In-flight (submitted but not yet completed) touch tasks. Prefetch requests are dropped when
  // this
  // exceeds maxPending, which bounds memory AND queue pressure under a sustained fault load without
  // the lock contention of a bounded blocking queue: keeping ~maxPending reads outstanding already
  // saturates the NVMe queue depth, so dropping the excess costs nothing (a missed prefetch is just
  // a
  // later synchronous fault). A dropped request is graceful degradation, never an error.
  private final AtomicLong pending = new AtomicLong();
  private final long maxPending;

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
    // A plain fixed pool (LinkedBlockingQueue, separate put/take locks). Memory and queue depth are
    // bounded by the pending-drop guard in prefetch(), not by a single-lock bounded queue — the
    // latter serialises a many-thread build on the queue lock. Keep the submit rate low by batching
    // a whole neighbor list into one task (see prefetch(int[], int)).
    this.ioPool = Executors.newFixedThreadPool(ioThreads, new PrefetchThreadFactory());
    this.maxPending = (long) ioThreads * 4;
    this.failureSink = failureSink;
    this.useSegments = vectors.supportsSegments();
    this.rawVectorBytes = vectors.dimension() * Float.BYTES;
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
    if (pending.get() >= maxPending) return; // I/O already saturated — drop (graceful)
    pending.incrementAndGet();
    submittedCount.incrementAndGet();
    ioPool.submit(
        () -> {
          try {
            touch(ordinal);
          } catch (Throwable t) {
            record(t);
          } finally {
            pending.decrementAndGet();
          }
        });
  }

  /**
   * Submits a single async touch for a whole neighbor list — one pool task touches all {@code
   * count} ordinals in {@code ordinals}. Batching keeps the submit rate ~1/degree of per-neighbor
   * prefetching, which is what makes prefetch viable during a many-threaded build: per-neighbor
   * submits at 10^8+ scale serialise every worker on the pool queue. The array is copied because
   * the caller reuses it after this returns.
   *
   * @param ordinals neighbor ordinals to prefetch
   * @param count number of valid entries at the front of {@code ordinals}
   */
  public void prefetch(int[] ordinals, int count) {
    if (count <= 0) return;
    if (pending.get() >= maxPending) return; // I/O already saturated — drop (graceful)
    int[] ids = Arrays.copyOf(ordinals, count);
    pending.incrementAndGet();
    submittedCount.incrementAndGet();
    ioPool.submit(
        () -> {
          try {
            for (int ord : ids) {
              touch(ord);
            }
          } catch (Throwable t) {
            record(t);
          } finally {
            pending.decrementAndGet();
          }
        });
  }

  /**
   * Pages in the vector at {@code ordinal} — zero-copy off the mmap slice when the store supports
   * it.
   */
  private void touch(int ordinal) {
    if (ordinal < 0 || ordinal >= vectors.size()) return;
    if (useSegments) {
      MemorySegment seg = vectors.vectorSegment(ordinal);
      for (long off = 0; off < rawVectorBytes; off += PAGE) {
        seg.get(ValueLayout.JAVA_BYTE, off);
      }
      seg.get(ValueLayout.JAVA_BYTE, rawVectorBytes - 1L);
    } else {
      vectors.getVector(ordinal); // in-RAM / non-segment: cheap array access, return discarded
    }
  }

  private void record(Throwable t) {
    failedCount.incrementAndGet();
    lastFailure.set(t);
    Consumer<Throwable> sink = failureSink;
    if (sink != null) {
      try {
        sink.accept(t);
      } catch (Throwable sinkFailure) {
        // A broken sink must not corrupt the pool thread; already counted via failedCount.
      }
    }
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
