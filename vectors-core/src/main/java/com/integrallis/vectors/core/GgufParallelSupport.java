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
import java.util.Objects;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

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
      positiveIntProperty("vectors.gguf.threads", PROCESSORS, PROCESSORS);
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

  static void forEachRange(
      MemorySegment firstWeights,
      MemorySegment secondWeights,
      MemorySegment thirdWeights,
      int workItems,
      int cols,
      GgufStagePlan.RangeOperation rangeOperation) {
    Objects.requireNonNull(firstWeights, "firstWeights");
    Objects.requireNonNull(secondWeights, "secondWeights");
    Objects.requireNonNull(thirdWeights, "thirdWeights");
    Objects.requireNonNull(rangeOperation, "rangeOperation");
    if (workItems < 0) {
      throw new IllegalArgumentException("workItems must not be negative: " + workItems);
    }
    if (workItems == 0) {
      return;
    }
    boolean shareable =
        firstWeights.isAccessibleBy(ACCESS_PROBE)
            && secondWeights.isAccessibleBy(ACCESS_PROBE)
            && thirdWeights.isAccessibleBy(ACCESS_PROBE);
    if (shareable && shouldParallelize(workItems, cols, PARALLELISM, ENABLED, MIN_ELEMENTS)) {
      ExecutorHolder.INSTANCE.execute(
          GgufStagePlan.of(GgufStagePlan.stage(workItems, rangeOperation)));
      return;
    }
    rangeOperation.execute(0, workItems);
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

  enum ExecutionMode {
    COMMON,
    DEDICATED,
    PERSISTENT;

    static ExecutionMode parse(String configured) {
      if (configured == null || configured.isBlank()) {
        return PERSISTENT;
      }
      try {
        return valueOf(configured.trim().toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException exception) {
        throw new IllegalArgumentException(
            "vectors.gguf.executor must be common, dedicated, or persistent: " + configured,
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
