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
import java.util.Locale;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/** Bounded row parallelism for large, thread-shareable GGUF matrices. */
final class GgufParallelSupport {

  static final long DEFAULT_MIN_ELEMENTS = 1_048_576L;
  static final long Q8_MIN_ELEMENTS = 4_194_304L;
  static final long PACKED_INT4_MIN_ELEMENTS = 65_536L;

  private static final boolean ENABLED =
      Boolean.parseBoolean(System.getProperty("vectors.gguf.parallel", "true"));
  private static final long MIN_ELEMENTS =
      Math.max(1L, Long.getLong("vectors.gguf.parallelThreshold", DEFAULT_MIN_ELEMENTS));
  private static final int PROCESSORS = Runtime.getRuntime().availableProcessors();

  /**
   * Default worker count: physical cores, not hardware threads.
   *
   * <p>This used to be {@code availableProcessors()}, which is the wrong quantity for a
   * compute-bound kernel and wrong by a factor of two on every SMT machine. Our own measurement
   * says so: batched decision prefill on a 4-physical-core box halves from one thread to two and
   * again to four and then saturates, at the core count rather than the processor count. Two
   * hardware threads sharing one core's vector units do not do two cores of matmul, but they do
   * each arrive at every barrier and contend for the same units.
   *
   * <p>{@code vectors.gguf.threads} still overrides it, and is still capped at the processor count:
   * the point is that the default is derived from the machine instead of waiting to be discovered.
   */
  private static final int PARALLELISM =
      positiveIntProperty("vectors.gguf.threads", CpuTopology.computeThreads(), PROCESSORS);

  private static final int CHUNKS_PER_THREAD =
      positiveIntProperty("vectors.gguf.chunksPerThread", 2, Integer.MAX_VALUE);
  private static final ExecutionMode EXECUTION_MODE =
      ExecutionMode.parse(System.getProperty("vectors.gguf.executor"));
  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  private GgufParallelSupport() {}

  static void forEachRow(MemorySegment weights, int rows, int cols, IntConsumer rowOperation) {
    forEachRow(weights, rows, cols, DEFAULT_MIN_ELEMENTS, rowOperation);
  }

  static void forEachRow(
      MemorySegment weights, int rows, int cols, long formatMinElements, IntConsumer rowOperation) {
    boolean shareable = weights.isAccessibleBy(ACCESS_PROBE);
    forEachRow(shareable, rows, cols, formatMinElements, rowOperation);
  }

  static void forEachPackedInt4Unit(
      MemorySegment packed,
      MemorySegment scales,
      int units,
      int workPerUnit,
      IntConsumer operation) {
    boolean shareable = packed.isAccessibleBy(ACCESS_PROBE) && scales.isAccessibleBy(ACCESS_PROBE);
    long minimum =
        Math.max(
            1L, Long.getLong("vectors.gguf.packedInt4ParallelThreshold", PACKED_INT4_MIN_ELEMENTS));
    if (shareable && shouldParallelize(units, workPerUnit, PARALLELISM, ENABLED, minimum)) {
      ExecutorHolder.INSTANCE.forEach(units, operation);
      return;
    }
    for (int unit = 0; unit < units; unit++) {
      operation.accept(unit);
    }
  }

  static void forEachRow(
      MemorySegment firstWeights,
      MemorySegment secondWeights,
      int rows,
      int cols,
      IntConsumer rowOperation) {
    forEachRow(firstWeights, secondWeights, rows, cols, DEFAULT_MIN_ELEMENTS, rowOperation);
  }

  static void forEachRow(
      MemorySegment firstWeights,
      MemorySegment secondWeights,
      int rows,
      int cols,
      long formatMinElements,
      IntConsumer rowOperation) {
    boolean shareable =
        firstWeights.isAccessibleBy(ACCESS_PROBE) && secondWeights.isAccessibleBy(ACCESS_PROBE);
    forEachRow(shareable, rows, cols, formatMinElements, rowOperation);
  }

  static void forEachRow(
      MemorySegment firstWeights,
      MemorySegment secondWeights,
      MemorySegment thirdWeights,
      int rows,
      int cols,
      IntConsumer rowOperation) {
    forEachRow(
        firstWeights, secondWeights, thirdWeights, rows, cols, DEFAULT_MIN_ELEMENTS, rowOperation);
  }

  static void forEachRow(
      MemorySegment firstWeights,
      MemorySegment secondWeights,
      MemorySegment thirdWeights,
      int rows,
      int cols,
      long formatMinElements,
      IntConsumer rowOperation) {
    boolean shareable =
        firstWeights.isAccessibleBy(ACCESS_PROBE)
            && secondWeights.isAccessibleBy(ACCESS_PROBE)
            && thirdWeights.isAccessibleBy(ACCESS_PROBE);
    forEachRow(shareable, rows, cols, formatMinElements, rowOperation);
  }

  private static void forEachRow(
      boolean shareable, int rows, int cols, long formatMinElements, IntConsumer rowOperation) {
    long effectiveMinElements = Math.max(MIN_ELEMENTS, formatMinElements);
    if (shareable && shouldParallelize(rows, cols, PARALLELISM, ENABLED, effectiveMinElements)) {
      ExecutorHolder.INSTANCE.forEach(rows, rowOperation);
      return;
    }
    for (int row = 0; row < rows; row++) {
      rowOperation.accept(row);
    }
  }

  /**
   * Whether a matrix operation over {@code elements} weight elements would be dispatched to the
   * shared executor under the same policy as {@link #forEachRow}.
   */
  static boolean willParallelize(MemorySegment weights, long elements, long formatMinElements) {
    long effectiveMinElements = Math.max(MIN_ELEMENTS, formatMinElements);
    return weights.isAccessibleBy(ACCESS_PROBE)
        && ENABLED
        && PARALLELISM > 1
        && elements >= effectiveMinElements;
  }

  /** Runs {@code tasks} on the shared executor when {@code parallel}, else inline in order. */
  static void forEachTask(boolean parallel, int tasks, IntConsumer operation) {
    if (parallel) {
      ExecutorHolder.INSTANCE.forEach(tasks, operation);
      return;
    }
    for (int task = 0; task < tasks; task++) {
      operation.accept(task);
    }
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

  static ExecutionMode executionMode() {
    return EXECUTION_MODE;
  }

  static int parallelism() {
    return PARALLELISM;
  }

  static int chunksPerThread() {
    return CHUNKS_PER_THREAD;
  }

  static void execute(GgufStagePlan plan) {
    if (ENABLED && PARALLELISM > 1) {
      ExecutorHolder.INSTANCE.execute(plan);
      return;
    }
    plan.executeSerially();
  }

  static GgufRowExecutor newExecutor(
      ExecutionMode mode, int parallelism, int chunksPerThread, String threadNamePrefix) {
    return switch (mode) {
      case COMMON ->
          (rows, rowOperation) -> IntStream.range(0, rows).parallel().forEach(rowOperation);
      case DEDICATED -> new GgufDedicatedRowExecutor(parallelism, threadNamePrefix);
      case PERSISTENT ->
          new GgufPersistentRowExecutor(parallelism, chunksPerThread, threadNamePrefix);
      case COUNTDOWN ->
          new GgufCountdownRowExecutor(parallelism, chunksPerThread, threadNamePrefix);
    };
  }

  private static int positiveIntProperty(String name, int defaultValue, int maximum) {
    String configured = System.getProperty(name);
    if (configured == null) {
      return defaultValue;
    }
    try {
      int value = Integer.parseInt(configured);
      if (value < 1) {
        throw new IllegalArgumentException(name + " must be positive: " + configured);
      }
      return Math.min(value, maximum);
    } catch (NumberFormatException exception) {
      throw new IllegalArgumentException(name + " must be an integer: " + configured, exception);
    }
  }

  /**
   * How a row-parallel job reaches the workers.
   *
   * <p>{@link #COUNTDOWN} is the default on measurement. On an 8 vCPU EPYC Milan box
   * (benchmark-results/2026-11-03-gguf-stage-barrier) it publishes a stage plan 2.59x faster than
   * {@link #PERSISTENT} at eight threads, and the reason is the scaling rather than the ratio:
   * going from one thread to eight costs the Phaser implementation 3.09x more per dispatch and this
   * one 1.28x, because a Phaser keeps the arrival count and the release signal in one word that
   * every party both CASes and polls. On the row path, which refuses to go parallel below a
   * mebibyte of weights, the two are indistinguishable -- a published null
   * (benchmark-results/2026-11-03-gguf-dispatch-barrier), since a 2 us handshake cannot show up
   * beside 450 us of arithmetic.
   *
   * <p>{@link #PERSISTENT} stays reachable as the other arm of that comparison until the end-to-end
   * token-rate A/B has run against it; it is a measurement seam, not a supported alternative.
   */
  enum ExecutionMode {
    COMMON,
    DEDICATED,
    PERSISTENT,
    COUNTDOWN;

    static ExecutionMode parse(String configured) {
      if (configured == null || configured.isBlank()) {
        return COUNTDOWN;
      }
      try {
        return valueOf(configured.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException exception) {
        throw new IllegalArgumentException(
            "vectors.gguf.executor must be common, dedicated, persistent or countdown: "
                + configured,
            exception);
      }
    }
  }

  private static final class ExecutorHolder {
    private static final GgufRowExecutor INSTANCE =
        newExecutor(EXECUTION_MODE, PARALLELISM, CHUNKS_PER_THREAD, "vectors-gguf");

    private ExecutorHolder() {}
  }
}
