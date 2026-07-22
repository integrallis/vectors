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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
  void executionModePropertyIsValidated() {
    assertThat(GgufParallelSupport.ExecutionMode.parse(null))
        .isEqualTo(GgufParallelSupport.ExecutionMode.PERSISTENT);
    assertThat(GgufParallelSupport.ExecutionMode.parse("common"))
        .isEqualTo(GgufParallelSupport.ExecutionMode.COMMON);
    assertThat(GgufParallelSupport.ExecutionMode.parse("dedicated"))
        .isEqualTo(GgufParallelSupport.ExecutionMode.DEDICATED);
    assertThat(GgufParallelSupport.ExecutionMode.parse("persistent"))
        .isEqualTo(GgufParallelSupport.ExecutionMode.PERSISTENT);
    assertThatThrownBy(() -> GgufParallelSupport.ExecutionMode.parse("unknown"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vectors.gguf.executor");
  }

  @Test
  void everyExecutionModeRunsEachRowExactlyOnce() {
    for (GgufParallelSupport.ExecutionMode mode : GgufParallelSupport.ExecutionMode.values()) {
      try (GgufRowExecutor executor = GgufParallelSupport.newExecutor(mode, 4, 4, "vectors-test")) {
        AtomicIntegerArray visits = new AtomicIntegerArray(257);
        executor.forEach(257, row -> visits.incrementAndGet(row));

        for (int row = 0; row < visits.length(); row++) {
          assertThat(visits.get(row)).as("%s row %s", mode, row).isOne();
        }
      }
    }
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
  void groupedRowsRemainOnOwnerWhenAnySegmentIsConfined() {
    AtomicInteger rowsVisited = new AtomicInteger();
    Set<Thread> threads = ConcurrentHashMap.newKeySet();
    Thread owner = Thread.currentThread();

    try (Arena arena = Arena.ofConfined()) {
      GgufParallelSupport.forEachRow(
          MemorySegment.ofArray(new byte[1]),
          MemorySegment.ofArray(new byte[1]),
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
  void groupedRangesRemainOnOwnerAndCoverEveryItemWhenAnySegmentIsConfined() {
    AtomicIntegerArray visits = new AtomicIntegerArray(1024);
    Set<Thread> threads = ConcurrentHashMap.newKeySet();
    Thread owner = Thread.currentThread();

    try (Arena arena = Arena.ofConfined()) {
      GgufParallelSupport.forEachRange(
          MemorySegment.ofArray(new byte[1]),
          MemorySegment.ofArray(new byte[1]),
          arena.allocate(1),
          visits.length(),
          2048,
          (fromInclusive, toExclusive) -> {
            threads.add(Thread.currentThread());
            for (int item = fromInclusive; item < toExclusive; item++) {
              visits.incrementAndGet(item);
            }
          });
    }

    for (int item = 0; item < visits.length(); item++) {
      assertThat(visits.get(item)).as("range item %s", item).isOne();
    }
    assertThat(threads).containsExactly(owner);
  }

  @Test
  void persistentExecutorRunsEveryRowExactlyOnceAcrossRepeatedDispatches() {
    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test")) {
      for (int round = 0; round < 100; round++) {
        int rows = 1 + round % 61;
        AtomicIntegerArray visits = new AtomicIntegerArray(rows);

        executor.forEach(rows, row -> visits.incrementAndGet(row));

        for (int row = 0; row < rows; row++) {
          assertThat(visits.get(row)).as("round %s row %s", round, row).isOne();
        }
      }
    }
  }

  @Test
  void persistentExecutorRunsOrderedStagesWithExactRangeCoverage() {
    AtomicInteger firstCompleted = new AtomicInteger();
    AtomicIntegerArray firstVisits = new AtomicIntegerArray(257);
    AtomicIntegerArray secondVisits = new AtomicIntegerArray(61);
    GgufStagePlan plan =
        GgufStagePlan.of(
            GgufStagePlan.stage(
                firstVisits.length(),
                (fromInclusive, toExclusive) -> {
                  for (int item = fromInclusive; item < toExclusive; item++) {
                    firstVisits.incrementAndGet(item);
                    firstCompleted.incrementAndGet();
                  }
                }),
            GgufStagePlan.stage(
                secondVisits.length(),
                (fromInclusive, toExclusive) -> {
                  if (firstCompleted.get() != firstVisits.length()) {
                    throw new AssertionError("second stage observed incomplete first stage");
                  }
                  for (int item = fromInclusive; item < toExclusive; item++) {
                    secondVisits.incrementAndGet(item);
                  }
                }));

    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test")) {
      executor.execute(plan);
    }

    for (int item = 0; item < firstVisits.length(); item++) {
      assertThat(firstVisits.get(item)).as("first stage item %s", item).isOne();
    }
    for (int item = 0; item < secondVisits.length(); item++) {
      assertThat(secondVisits.get(item)).as("second stage item %s", item).isOne();
    }
  }

  @Test
  void persistentExecutorPropagatesStageFailureAndRemainsReusable() {
    GgufStagePlan failing =
        GgufStagePlan.of(
            GgufStagePlan.stage(64, (fromInclusive, toExclusive) -> {}),
            GgufStagePlan.stage(
                64,
                (fromInclusive, toExclusive) -> {
                  if (fromInclusive <= 17 && 17 < toExclusive) {
                    throw new IllegalStateException("stage boom");
                  }
                }));

    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test")) {
      assertThatThrownBy(() -> executor.execute(failing))
          .isInstanceOf(IllegalStateException.class)
          .hasMessage("stage boom");

      AtomicIntegerArray visits = new AtomicIntegerArray(64);
      executor.execute(
          GgufStagePlan.of(
              GgufStagePlan.stage(
                  visits.length(),
                  (fromInclusive, toExclusive) -> {
                    for (int item = fromInclusive; item < toExclusive; item++) {
                      visits.incrementAndGet(item);
                    }
                  })));
      for (int item = 0; item < visits.length(); item++) {
        assertThat(visits.get(item)).as("reused item %s", item).isOne();
      }
    }
  }

  @Test
  void persistentExecutorSerializesConcurrentStagePublishers() throws Exception {
    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test");
        ExecutorService publishers = Executors.newFixedThreadPool(4)) {
      List<Future<AtomicIntegerArray>> results = new ArrayList<>();
      for (int publisher = 0; publisher < 4; publisher++) {
        results.add(
            publishers.submit(
                () -> {
                  AtomicIntegerArray visits = new AtomicIntegerArray(97);
                  GgufStagePlan plan =
                      GgufStagePlan.of(
                          GgufStagePlan.stage(
                              visits.length(),
                              (fromInclusive, toExclusive) -> {
                                for (int item = fromInclusive; item < toExclusive; item++) {
                                  visits.incrementAndGet(item);
                                }
                              }));
                  for (int round = 0; round < 20; round++) {
                    executor.execute(plan);
                  }
                  return visits;
                }));
      }

      for (Future<AtomicIntegerArray> result : results) {
        AtomicIntegerArray visits = result.get();
        for (int item = 0; item < visits.length(); item++) {
          assertThat(visits.get(item)).as("stage item %s", item).isEqualTo(20);
        }
      }
    }
  }

  @Test
  void persistentExecutorSerializesConcurrentPublishersWithoutLosingRows() throws Exception {
    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test");
        ExecutorService publishers = Executors.newFixedThreadPool(4)) {
      List<Future<AtomicIntegerArray>> results = new ArrayList<>();
      for (int publisher = 0; publisher < 4; publisher++) {
        results.add(
            publishers.submit(
                () -> {
                  AtomicIntegerArray visits = new AtomicIntegerArray(97);
                  for (int round = 0; round < 20; round++) {
                    executor.forEach(97, row -> visits.incrementAndGet(row));
                  }
                  return visits;
                }));
      }

      for (Future<AtomicIntegerArray> result : results) {
        AtomicIntegerArray visits = result.get();
        for (int row = 0; row < visits.length(); row++) {
          assertThat(visits.get(row)).as("row %s", row).isEqualTo(20);
        }
      }
    }
  }

  @Test
  void persistentExecutorPropagatesWorkerFailureAndRemainsReusable() {
    try (GgufPersistentRowExecutor executor = new GgufPersistentRowExecutor(4, 4, "vectors-test")) {
      assertThatThrownBy(
              () ->
                  executor.forEach(
                      64,
                      row -> {
                        if (row == 17) {
                          throw new IllegalStateException("boom");
                        }
                      }))
          .isInstanceOf(IllegalStateException.class)
          .hasMessage("boom");

      AtomicInteger rowsVisited = new AtomicInteger();
      executor.forEach(64, ignored -> rowsVisited.incrementAndGet());
      assertThat(rowsVisited).hasValue(64);
    }
  }

  @Test
  void persistentExecutorClosesImmediatelyAfterACompletedDispatch() {
    assertTimeoutPreemptively(
        Duration.ofSeconds(5),
        () -> {
          for (int attempt = 0; attempt < 100; attempt++) {
            try (GgufPersistentRowExecutor executor =
                new GgufPersistentRowExecutor(12, 4, "vectors-test")) {
              executor.forEach(257, ignored -> Thread.onSpinWait());
            }
          }
        });
  }

  @Test
  void persistentExecutorHandlesMaximumChunkConfigurationWithoutOverflow() {
    assertTimeoutPreemptively(
        Duration.ofSeconds(5),
        () -> {
          try (GgufPersistentRowExecutor executor =
              new GgufPersistentRowExecutor(4, Integer.MAX_VALUE, "vectors-test")) {
            AtomicInteger rowsVisited = new AtomicInteger();
            executor.forEach(257, ignored -> rowsVisited.incrementAndGet());
            assertThat(rowsVisited).hasValue(257);
          }
        });
  }
}
