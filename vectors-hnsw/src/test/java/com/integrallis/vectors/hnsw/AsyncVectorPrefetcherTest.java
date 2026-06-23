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

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Pins the observability contract of {@link AsyncVectorPrefetcher}: in-range touches succeed and
 * are counted, out-of-range ordinals are not submitted, and failures during the touch are recorded
 * in an observable counter plus delivered to an optional failure sink.
 *
 * <p>Regression coverage for the audit finding "AsyncVectorPrefetcher swallows all Exception — SSD
 * page-fault errors vanish silently." The fix surfaces failures without changing the
 * fire-and-forget contract.
 */
class AsyncVectorPrefetcherTest {

  @Test
  void inRangeOrdinal_isSubmittedAndCounted() throws Exception {
    CountingVectors v = new CountingVectors(4, 2);
    try (AsyncVectorPrefetcher p = new AsyncVectorPrefetcher(v, 2)) {
      p.prefetch(2);
      assertThat(p.submittedCount()).isEqualTo(1);
      v.awaitInvocations(1, Duration.ofSeconds(5));
      assertThat(v.invocations.get()).isEqualTo(1);
      assertThat(p.failedCount()).isZero();
      assertThat(p.lastFailure()).isNull();
    }
  }

  @Test
  void outOfRangeOrdinal_isNotSubmitted() {
    CountingVectors v = new CountingVectors(4, 2);
    try (AsyncVectorPrefetcher p = new AsyncVectorPrefetcher(v, 2)) {
      p.prefetch(-1);
      p.prefetch(99);
      assertThat(p.submittedCount()).isZero();
      assertThat(v.invocations.get()).isZero();
      assertThat(p.failedCount()).isZero();
    }
  }

  @Test
  void touchFailure_isObservableViaCounterAndSink() throws Exception {
    RuntimeException boom = new RuntimeException("simulated SSD page fault");
    ThrowingVectors v = new ThrowingVectors(4, 2, boom);
    AtomicInteger sinkCalls = new AtomicInteger();
    AtomicReference<Throwable> sinkLast = new AtomicReference<>();
    try (AsyncVectorPrefetcher p =
        new AsyncVectorPrefetcher(
            v,
            1,
            t -> {
              sinkLast.set(t);
              sinkCalls.incrementAndGet();
            })) {
      p.prefetch(0);
      v.awaitInvocations(1, Duration.ofSeconds(5));
      awaitFailures(p, 1, Duration.ofSeconds(5));
      assertThat(p.submittedCount()).isEqualTo(1);
      assertThat(p.failedCount()).isEqualTo(1);
      assertThat(p.lastFailure()).isSameAs(boom);
      assertThat(sinkCalls.get()).isEqualTo(1);
      assertThat(sinkLast.get()).isSameAs(boom);
    }
  }

  @Test
  void failingSinkItselfDoesNotPropagateOrCorruptCounters() throws Exception {
    ThrowingVectors v = new ThrowingVectors(4, 2, new RuntimeException("touch fail"));
    try (AsyncVectorPrefetcher p =
        new AsyncVectorPrefetcher(
            v,
            1,
            t -> {
              throw new RuntimeException("sink fail");
            })) {
      p.prefetch(0);
      v.awaitInvocations(1, Duration.ofSeconds(5));
      awaitFailures(p, 1, Duration.ofSeconds(5));
      assertThat(p.failedCount()).isEqualTo(1);
    }
  }

  private static void awaitFailures(AsyncVectorPrefetcher p, long target, Duration max)
      throws InterruptedException {
    long deadline = System.nanoTime() + max.toNanos();
    while (System.nanoTime() < deadline) {
      if (p.failedCount() >= target) return;
      Thread.sleep(20);
    }
    throw new AssertionError(
        "prefetcher.failedCount did not reach " + target + "; got " + p.failedCount());
  }

  private static final class CountingVectors implements RandomAccessVectors {
    private final int size;
    private final int dim;
    final AtomicInteger invocations = new AtomicInteger();
    private final CountDownLatch firstInvocation = new CountDownLatch(1);

    CountingVectors(int size, int dim) {
      this.size = size;
      this.dim = dim;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public float[] getVector(int ordinal) {
      invocations.incrementAndGet();
      firstInvocation.countDown();
      return new float[dim];
    }

    void awaitInvocations(int target, Duration max) throws InterruptedException {
      firstInvocation.await(max.toMillis(), TimeUnit.MILLISECONDS);
      long deadline = System.nanoTime() + max.toNanos();
      while (System.nanoTime() < deadline) {
        if (invocations.get() >= target) return;
        Thread.sleep(10);
      }
      throw new AssertionError(
          "vectors.invocations did not reach " + target + "; got " + invocations.get());
    }
  }

  private static final class ThrowingVectors implements RandomAccessVectors {
    private final int size;
    private final int dim;
    private final RuntimeException toThrow;
    final AtomicInteger invocations = new AtomicInteger();
    private final CountDownLatch firstInvocation = new CountDownLatch(1);

    ThrowingVectors(int size, int dim, RuntimeException toThrow) {
      this.size = size;
      this.dim = dim;
      this.toThrow = toThrow;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public float[] getVector(int ordinal) {
      invocations.incrementAndGet();
      firstInvocation.countDown();
      throw toThrow;
    }

    void awaitInvocations(int target, Duration max) throws InterruptedException {
      firstInvocation.await(max.toMillis(), TimeUnit.MILLISECONDS);
      long deadline = System.nanoTime() + max.toNanos();
      while (System.nanoTime() < deadline) {
        if (invocations.get() >= target) return;
        Thread.sleep(10);
      }
      throw new AssertionError(
          "vectors.invocations did not reach " + target + "; got " + invocations.get());
    }
  }
}
