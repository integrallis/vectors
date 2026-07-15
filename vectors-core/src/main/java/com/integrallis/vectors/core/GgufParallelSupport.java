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
  private static final Thread ACCESS_PROBE = Thread.ofPlatform().unstarted(() -> {});

  private GgufParallelSupport() {}

  static void forEachRow(MemorySegment weights, int rows, int cols, IntConsumer rowOperation) {
    forEachRow(weights, rows, cols, DEFAULT_MIN_ELEMENTS, rowOperation);
  }

  static void forEachRow(
      MemorySegment weights, int rows, int cols, long formatMinElements, IntConsumer rowOperation) {
    boolean shareable = weights.isAccessibleBy(ACCESS_PROBE);
    long effectiveMinElements = Math.max(MIN_ELEMENTS, formatMinElements);
    if (shareable && shouldParallelize(rows, cols, PROCESSORS, ENABLED, effectiveMinElements)) {
      IntStream.range(0, rows).parallel().forEach(rowOperation);
      return;
    }
    for (int row = 0; row < rows; row++) {
      rowOperation.accept(row);
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
}
