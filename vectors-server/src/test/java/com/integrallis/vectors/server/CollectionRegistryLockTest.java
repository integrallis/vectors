/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins the contract documented on {@code DocumentsRoutes}: write paths must run "under the
 * collection's per-name write lock". The lock is owned by {@link CollectionRegistry} (the same
 * {@code nameLocks} map already used by {@code create}/{@code drop}); {@code runUnderWriteLock} is
 * the public surface that exposes it to write routes so a batch's upsert + commit pair is
 * observably atomic w.r.t. other batches against the same collection.
 *
 * <p>Same-name calls must serialise (only one runnable may execute at a time per collection).
 * Different-name calls must not block each other.
 */
@Tag("unit")
class CollectionRegistryLockTest {

  @Test
  void runUnderWriteLock_serialisesSameName() throws Exception {
    CollectionRegistry registry = new CollectionRegistry();
    int callers = 8;
    AtomicInteger inside = new AtomicInteger();
    AtomicInteger maxConcurrent = new AtomicInteger();
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(callers);

    try (var pool = Executors.newFixedThreadPool(callers)) {
      for (int i = 0; i < callers; i++) {
        pool.submit(
            () -> {
              try {
                start.await();
                registry.runUnderWriteLock(
                    "c1",
                    () -> {
                      int now = inside.incrementAndGet();
                      maxConcurrent.updateAndGet(prev -> Math.max(prev, now));
                      // Hold briefly so any concurrent entry has a window to be observed.
                      try {
                        Thread.sleep(15);
                      } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                      }
                      inside.decrementAndGet();
                    });
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              } finally {
                done.countDown();
              }
            });
      }
      start.countDown();
      assertThat(done.await(10, TimeUnit.SECONDS))
          .as("all runnables must complete within budget")
          .isTrue();
    }

    assertThat(maxConcurrent.get())
        .as("at most one runnable per collection name may execute concurrently")
        .isEqualTo(1);
  }

  @Test
  void runUnderWriteLock_doesNotSerialiseDifferentNames() throws Exception {
    CollectionRegistry registry = new CollectionRegistry();
    CountDownLatch firstEntered = new CountDownLatch(1);
    CountDownLatch releaseFirst = new CountDownLatch(1);
    CountDownLatch bothDone = new CountDownLatch(2);

    try (var pool = Executors.newFixedThreadPool(2)) {
      pool.submit(
          () -> {
            try {
              registry.runUnderWriteLock(
                  "a",
                  () -> {
                    firstEntered.countDown();
                    try {
                      // Hold the "a" lock until the second caller signals it succeeded.
                      releaseFirst.await();
                    } catch (InterruptedException e) {
                      Thread.currentThread().interrupt();
                    }
                  });
            } finally {
              bothDone.countDown();
            }
          });

      pool.submit(
          () -> {
            try {
              firstEntered.await();
              // The "a" lock is held; the "b" lock must not block on it.
              registry.runUnderWriteLock("b", () -> {});
              releaseFirst.countDown();
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
            } finally {
              bothDone.countDown();
            }
          });

      assertThat(bothDone.await(5, TimeUnit.SECONDS))
          .as("different-name lock must not block on another name")
          .isTrue();
    }
  }

  @Test
  void runUnderWriteLock_propagatesRunnableException() {
    CollectionRegistry registry = new CollectionRegistry();
    org.junit.jupiter.api.Assertions.assertThrows(
        IllegalStateException.class,
        () ->
            registry.runUnderWriteLock(
                "c",
                () -> {
                  throw new IllegalStateException("boom");
                }));
    // And the lock must be released, so a subsequent call on the same name succeeds.
    registry.runUnderWriteLock("c", () -> {});
  }
}
