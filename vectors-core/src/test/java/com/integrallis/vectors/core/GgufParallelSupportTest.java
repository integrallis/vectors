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

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.Arena;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufParallelSupportTest {

  @Test
  void parallelizesOnlyLargeMatricesWhenMultipleProcessorsAreAvailable() {
    assertThat(GgufParallelSupport.shouldParallelize(1024, 2048, 8, true, 1_048_576L)).isTrue();
    assertThat(GgufParallelSupport.shouldParallelize(32, 2048, 8, true, 1_048_576L)).isFalse();
    assertThat(GgufParallelSupport.shouldParallelize(1024, 2048, 1, true, 1_048_576L)).isFalse();
    assertThat(GgufParallelSupport.shouldParallelize(1024, 2048, 8, false, 1_048_576L)).isFalse();
  }

  @Test
  void workCalculationDoesNotOverflowIntRange() {
    assertThat(
            GgufParallelSupport.shouldParallelize(
                Integer.MAX_VALUE, Integer.MAX_VALUE, 8, true, Integer.MAX_VALUE))
        .isTrue();
  }

  @Test
  void q8ThresholdKeepsSmallMatricesSerialAndLargeMatricesParallel() {
    assertThat(
            GgufParallelSupport.shouldParallelize(
                1024, 2048, 8, true, GgufParallelSupport.Q8_MIN_ELEMENTS))
        .isFalse();
    assertThat(
            GgufParallelSupport.shouldParallelize(
                8192, 2048, 8, true, GgufParallelSupport.Q8_MIN_ELEMENTS))
        .isTrue();
  }

  @Test
  void confinedSegmentsRemainOnTheirOwnerThread() {
    AtomicInteger rowsVisited = new AtomicInteger();
    Set<Thread> threads = ConcurrentHashMap.newKeySet();
    Thread owner = Thread.currentThread();

    try (Arena arena = Arena.ofConfined()) {
      GgufParallelSupport.forEachRow(
          arena.allocate(1),
          1024,
          2048,
          ignored -> {
            rowsVisited.incrementAndGet();
            threads.add(Thread.currentThread());
          });
    }

    assertThat(rowsVisited).hasValue(1024);
    assertThat(threads).containsExactly(owner);
  }

  @Test
  void boundedParallelExecutionVisitsEveryRowExactlyOnce() {
    int rows = 257;
    AtomicIntegerArray visits = new AtomicIntegerArray(rows);
    Set<Thread> threads = ConcurrentHashMap.newKeySet();

    GgufParallelSupport.forEachRowParallel(
        rows,
        3,
        row -> {
          visits.incrementAndGet(row);
          threads.add(Thread.currentThread());
        });

    for (int row = 0; row < rows; row++) {
      assertThat(visits.get(row)).isOne();
    }
    assertThat(threads).hasSizeBetween(1, 3);
  }
}
