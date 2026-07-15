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

import java.lang.foreign.MemorySegment;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

/** Bounded row parallelism for large, thread-shareable GGUF matrices. */
final class GgufParallelSupport {

  static final long DEFAULT_MIN_ELEMENTS = 1_048_576L;
  static final long Q8_MIN_ELEMENTS = 4_194_304L;

  private static final boolean ENABLED =
      Boolean.parseBoolean(System.getProperty("vectors.gguf.parallel", "true"));
  private static final long MIN_ELEMENTS =
      Math.max(1L, Long.getLong("vectors.gguf.parallelThreshold", DEFAULT_MIN_ELEMENTS));
  private static final int PROCESSORS = Runtime.getRuntime().availableProcessors();
  private static final int PARALLELISM =
      Math.max(1, Math.min(PROCESSORS, Integer.getInteger("vectors.gguf.parallelism", PROCESSORS)));
  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});
  private static final AtomicInteger WORKER_SEQUENCE = new AtomicInteger();
  private static final ForkJoinPool ROW_POOL =
      new ForkJoinPool(
          PARALLELISM,
          pool -> {
            var worker = ForkJoinPool.defaultForkJoinWorkerThreadFactory.newThread(pool);
            worker.setName("vectors-gguf-" + WORKER_SEQUENCE.incrementAndGet());
            return worker;
          },
          null,
          false);

  private GgufParallelSupport() {}

  static void forEachRow(MemorySegment weights, int rows, int cols, IntConsumer rowOperation) {
    forEachRow(weights, rows, cols, DEFAULT_MIN_ELEMENTS, rowOperation);
  }

  static void forEachRow(
      MemorySegment weights, int rows, int cols, long formatMinElements, IntConsumer rowOperation) {
    boolean shareable = weights.isAccessibleBy(ACCESS_PROBE);
    long effectiveMinElements = Math.max(MIN_ELEMENTS, formatMinElements);
    if (shareable && shouldParallelize(rows, cols, PARALLELISM, ENABLED, effectiveMinElements)) {
      forEachRowInPool(ROW_POOL, rows, rowOperation);
      return;
    }
    for (int row = 0; row < rows; row++) {
      rowOperation.accept(row);
    }
  }

  static void forEachRowInPool(ForkJoinPool pool, int rows, IntConsumer rowOperation) {
    pool.invoke(new RowRangeAction(rows, rowOperation, pool.getParallelism()));
  }

  static boolean shouldParallelize(
      int rows, int cols, int processors, boolean enabled, long minElements) {
    return enabled && processors > 1 && (long) rows * cols >= minElements;
  }

  static boolean enabled() {
    return ENABLED;
  }

  static long minElements() {
    return MIN_ELEMENTS;
  }

  static int parallelism() {
    return PARALLELISM;
  }

  @SuppressWarnings("serial")
  private static final class RowRangeAction extends RecursiveAction {
    private final int rows;
    private final IntConsumer rowOperation;
    private final int parallelism;

    private RowRangeAction(int rows, IntConsumer rowOperation, int parallelism) {
      this.rows = rows;
      this.rowOperation = rowOperation;
      this.parallelism = parallelism;
    }

    @Override
    protected void compute() {
      int taskCount = Math.min(rows, parallelism);
      int rowsPerTask = (rows + taskCount - 1) / taskCount;
      RowBatchAction[] forked = new RowBatchAction[taskCount - 1];
      for (int task = 1; task < taskCount; task++) {
        int start = task * rowsPerTask;
        int end = Math.min(rows, start + rowsPerTask);
        RowBatchAction action = new RowBatchAction(start, end, rowOperation);
        forked[task - 1] = action;
        action.fork();
      }

      runRows(0, Math.min(rows, rowsPerTask), rowOperation);
      for (RowBatchAction action : forked) {
        action.join();
      }
    }
  }

  @SuppressWarnings("serial")
  private static final class RowBatchAction extends RecursiveAction {
    private final int start;
    private final int end;
    private final IntConsumer rowOperation;

    private RowBatchAction(int start, int end, IntConsumer rowOperation) {
      this.start = start;
      this.end = end;
      this.rowOperation = rowOperation;
    }

    @Override
    protected void compute() {
      runRows(start, end, rowOperation);
    }
  }

  private static void runRows(int start, int end, IntConsumer rowOperation) {
    for (int row = start; row < end; row++) {
      rowOperation.accept(row);
    }
  }
}
