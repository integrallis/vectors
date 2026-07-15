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

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/** A dedicated fork-join pool that isolates GGUF work from the JVM common pool. */
final class GgufDedicatedRowExecutor implements GgufRowExecutor {

  private final ForkJoinPool pool;

  GgufDedicatedRowExecutor(int parallelism, String threadNamePrefix) {
    if (parallelism < 1) {
      throw new IllegalArgumentException("parallelism must be positive");
    }
    AtomicInteger workerIds = new AtomicInteger();
    ForkJoinPool.ForkJoinWorkerThreadFactory workerFactory =
        owner -> newWorker(owner, threadNamePrefix, workerIds.getAndIncrement());
    pool = new ForkJoinPool(parallelism, workerFactory, null, true);
  }

  @Override
  public void forEach(int rows, IntConsumer rowOperation) {
    pool.submit(() -> IntStream.range(0, rows).parallel().forEach(rowOperation)).join();
  }

  private static ForkJoinWorkerThread newWorker(
      ForkJoinPool owner, String threadNamePrefix, int workerId) {
    ForkJoinWorkerThread worker = ForkJoinPool.defaultForkJoinWorkerThreadFactory.newThread(owner);
    worker.setName(threadNamePrefix + '-' + workerId);
    return worker;
  }

  @Override
  public void close() {
    pool.shutdown();
  }
}
