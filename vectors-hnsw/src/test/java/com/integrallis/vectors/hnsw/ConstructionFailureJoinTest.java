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

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.concurrent.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ConstructionFailureJoinTest {
  @Test
  void interruptionStillJoinsWorkersAndRestoresInterruptStatus() {
    var calls = new java.util.concurrent.atomic.AtomicInteger();
    CompletableFuture<Void> interruptedOnce =
        new CompletableFuture<>() {
          @Override
          public Void get() throws InterruptedException {
            if (calls.getAndIncrement() == 0) throw new InterruptedException("injected");
            return null;
          }
        };
    try {
      RuntimeException failure =
          assertThrows(RuntimeException.class, () -> await(List.of(interruptedOnce)));
      assertInstanceOf(InterruptedException.class, failure.getCause());
      assertEquals(2, calls.get());
      assertTrue(Thread.currentThread().isInterrupted());
    } finally {
      Thread.interrupted();
    }
  }

  @Test
  void workerFailureDoesNotReturnWhileAnotherWorkerCanStillMutateTheGraph() throws Exception {
    CountDownLatch joined = new CountDownLatch(1);
    CompletableFuture<Void> unfinished =
        new CompletableFuture<>() {
          @Override
          public Void get() throws InterruptedException, ExecutionException {
            joined.countDown();
            return super.get();
          }
        };
    IllegalStateException original = new IllegalStateException("injected worker failure");
    try (var caller = Executors.newSingleThreadExecutor()) {
      Future<?> result =
          caller.submit(() -> await(List.of(CompletableFuture.failedFuture(original), unfinished)));
      try {
        assertTrue(joined.await(2, TimeUnit.SECONDS), "must join remaining worker after a failure");
        assertFalse(result.isDone());
      } finally {
        unfinished.complete(null);
      }
      ExecutionException failure =
          assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
      assertSame(original, failure.getCause().getCause());
    }
  }

  private static void await(List<Future<?>> futures) {
    try {
      var method = ConcurrentHnswGraphBuilder.class.getDeclaredMethod("awaitAll", List.class);
      method.setAccessible(true);
      method.invoke(null, futures);
    } catch (InvocationTargetException e) {
      throw (RuntimeException) e.getCause();
    } catch (ReflectiveOperationException e) {
      throw new AssertionError(e);
    }
  }
}
