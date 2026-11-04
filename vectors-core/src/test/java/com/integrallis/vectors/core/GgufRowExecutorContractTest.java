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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.function.IntConsumer;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The behaviour every persistent GGUF executor owes its callers, asserted against each
 * implementation.
 *
 * <p>{@link GgufPersistentRowExecutor} and {@link GgufCountdownRowExecutor} differ only in how a
 * job is published and completion observed, so a performance comparison between them is only worth
 * reading if both satisfy the same contract. Running one parameterized suite over both is what
 * makes that true by construction rather than by having remembered to copy a test across.
 */
class GgufRowExecutorContractTest {

  /** A named factory, so a failure reports which implementation broke rather than a lambda. */
  private record Implementation(String name, ExecutorFactory factory) {

    @Override
    public String toString() {
      return name;
    }
  }

  @FunctionalInterface
  private interface ExecutorFactory {

    GgufRowExecutor create(int parallelism, int chunksPerWorker, String prefix);
  }

  private static Stream<Arguments> implementations() {
    return Stream.of(
        Arguments.of(new Implementation("persistent-phaser", GgufPersistentRowExecutor::new)),
        Arguments.of(new Implementation("countdown-generation", GgufCountdownRowExecutor::new)));
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void runsEveryRowExactlyOnceAcrossRepeatedDispatches(Implementation implementation) {
    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
      for (int round = 0; round < 100; round++) {
        int rows = 1 + round % 61;
        AtomicIntegerArray visits = new AtomicIntegerArray(rows);

        executor.forEach(rows, visits::incrementAndGet);

        for (int row = 0; row < rows; row++) {
          assertThat(visits.get(row)).as("round %s row %s", round, row).isOne();
        }
      }
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void runsOrderedStagesWithExactRangeCoverage(Implementation implementation) {
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

    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
      for (int round = 0; round < 25; round++) {
        firstCompleted.set(0);
        executor.execute(plan);
      }
    }

    for (int item = 0; item < firstVisits.length(); item++) {
      assertThat(firstVisits.get(item)).as("first stage item %s", item).isEqualTo(25);
    }
    for (int item = 0; item < secondVisits.length(); item++) {
      assertThat(secondVisits.get(item)).as("second stage item %s", item).isEqualTo(25);
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void propagatesWorkerFailureAndRemainsReusable(Implementation implementation) {
    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
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

  @ParameterizedTest
  @MethodSource("implementations")
  void propagatesStageFailureAndRemainsReusable(Implementation implementation) {
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

    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
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

  @ParameterizedTest
  @MethodSource("implementations")
  void serializesConcurrentPublishersWithoutLosingRows(Implementation implementation)
      throws Exception {
    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract");
        ExecutorService publishers = Executors.newFixedThreadPool(4)) {
      List<Future<AtomicIntegerArray>> results = new ArrayList<>();
      for (int publisher = 0; publisher < 4; publisher++) {
        results.add(
            publishers.submit(
                () -> {
                  AtomicIntegerArray visits = new AtomicIntegerArray(97);
                  for (int round = 0; round < 20; round++) {
                    executor.forEach(97, visits::incrementAndGet);
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

  @ParameterizedTest
  @MethodSource("implementations")
  void serializesConcurrentStagePublishers(Implementation implementation) throws Exception {
    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract");
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

  @ParameterizedTest
  @MethodSource("implementations")
  void nestedDispatchStaysOnTheCallingThread(Implementation implementation) {
    try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
      AtomicInteger nestedRows = new AtomicInteger();
      AtomicIntegerArray outerVisits = new AtomicIntegerArray(64);
      executor.forEach(
          64,
          row -> {
            outerVisits.incrementAndGet(row);
            Thread owner = Thread.currentThread();
            executor.forEach(
                3,
                nested -> {
                  if (Thread.currentThread() != owner) {
                    throw new AssertionError("nested dispatch left its calling thread");
                  }
                  nestedRows.incrementAndGet();
                });
          });

      for (int row = 0; row < outerVisits.length(); row++) {
        assertThat(outerVisits.get(row)).as("outer row %s", row).isOne();
      }
      assertThat(nestedRows).hasValue(64 * 3);
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void nestedDispatchAcrossTwoPoolsStaysOnTheCallingThread(Implementation implementation) {
    // The guard has to be shared between pools: were the inner pool to publish, its workers could
    // re-enter the outer one and the two would wait on each other.
    try (GgufRowExecutor outer = implementation.factory().create(4, 4, "contract-outer");
        GgufRowExecutor inner = implementation.factory().create(4, 4, "contract-inner")) {
      AtomicInteger nestedRows = new AtomicInteger();
      outer.forEach(
          32,
          row -> {
            Thread owner = Thread.currentThread();
            inner.forEach(
                5,
                nested -> {
                  if (Thread.currentThread() != owner) {
                    throw new AssertionError("cross-pool nested dispatch left its thread");
                  }
                  nestedRows.incrementAndGet();
                });
          });
      assertThat(nestedRows).hasValue(32 * 5);
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void closingFromInsideACallbackIsRejected(Implementation implementation) {
    // Not try-with-resources: the point of the test is the explicit close from inside a callback.
    GgufRowExecutor executor = implementation.factory().create(2, 1, "contract");
    try {
      assertThatThrownBy(() -> executor.forEach(8, row -> executor.close()))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("callback");
    } finally {
      executor.close();
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void closesImmediatelyAfterACompletedDispatch(Implementation implementation) {
    GgufRowExecutor executor = implementation.factory().create(4, 4, "contract");
    AtomicInteger rowsVisited = new AtomicInteger();
    executor.forEach(1024, ignored -> rowsVisited.incrementAndGet());
    executor.close();
    assertThat(rowsVisited).hasValue(1024);
    assertThatThrownBy(() -> executor.forEach(8, ignored -> {}))
        .isInstanceOf(IllegalStateException.class);
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void handlesMaximumChunkConfigurationWithoutOverflow(Implementation implementation) {
    try (GgufRowExecutor executor =
        implementation.factory().create(4, Integer.MAX_VALUE, "contract")) {
      AtomicIntegerArray visits = new AtomicIntegerArray(37);
      executor.forEach(37, visits::incrementAndGet);
      for (int row = 0; row < visits.length(); row++) {
        assertThat(visits.get(row)).as("row %s", row).isOne();
      }
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void parksAndResumesWhenThePollBudgetIsZero(Implementation implementation) {
    long original = GgufPersistentRowExecutor.pollMillis();
    try {
      GgufPersistentRowExecutor.setPollMillis(0);
      try (GgufRowExecutor executor = implementation.factory().create(4, 4, "contract")) {
        for (int round = 0; round < 20; round++) {
          AtomicIntegerArray visits = new AtomicIntegerArray(129);
          executor.forEach(129, visits::incrementAndGet);
          for (int row = 0; row < visits.length(); row++) {
            assertThat(visits.get(row)).as("round %s row %s", round, row).isOne();
          }
        }
      }
    } finally {
      GgufPersistentRowExecutor.setPollMillis(original);
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void runsEveryRowWhenWorkersOutnumberProcessors(Implementation implementation) {
    // The oversubscribed case: more parties than cores, where a poller can hold the core the
    // publisher needs. This is the shape that hangs if the barrier never yields.
    try (GgufRowExecutor executor = implementation.factory().create(12, 4, "contract")) {
      for (int round = 0; round < 25; round++) {
        AtomicIntegerArray visits = new AtomicIntegerArray(1031);
        executor.forEach(1031, visits::incrementAndGet);
        for (int row = 0; row < visits.length(); row++) {
          assertThat(visits.get(row)).as("round %s row %s", round, row).isOne();
        }
      }
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void soaksMixedDispatchesWhileWorkersParkAndResume(Implementation implementation) {
    // The park path is the delicate one: a participant that goes to sleep between the moment
    // completion is signalled and the moment it checks for it must still be woken, and the same
    // handshake has to survive a job type changing under it. Alternating forEach with a two-stage
    // plan at a zero poll budget drives every party through park-and-wake on every dispatch, which
    // a steady stream of same-shaped dispatches never does.
    long original = GgufPersistentRowExecutor.pollMillis();
    try {
      try (GgufRowExecutor executor = implementation.factory().create(6, 3, "soak")) {
        for (int round = 0; round < 120; round++) {
          GgufPersistentRowExecutor.setPollMillis(round % 2 == 0 ? 0 : 5);
          int rows = 1 + (round * 37) % 211;
          if (round % 3 == 2) {
            AtomicIntegerArray first = new AtomicIntegerArray(rows);
            AtomicIntegerArray second = new AtomicIntegerArray(1 + rows / 2);
            AtomicInteger firstDone = new AtomicInteger();
            executor.execute(
                GgufStagePlan.of(
                    GgufStagePlan.stage(
                        first.length(),
                        (from, to) -> {
                          for (int item = from; item < to; item++) {
                            first.incrementAndGet(item);
                            firstDone.incrementAndGet();
                          }
                        }),
                    GgufStagePlan.stage(
                        second.length(),
                        (from, to) -> {
                          if (firstDone.get() != first.length()) {
                            throw new AssertionError("stage two saw an incomplete stage one");
                          }
                          for (int item = from; item < to; item++) {
                            second.incrementAndGet(item);
                          }
                        })));
            for (int item = 0; item < first.length(); item++) {
              assertThat(first.get(item)).as("round %s stage-one %s", round, item).isOne();
            }
            for (int item = 0; item < second.length(); item++) {
              assertThat(second.get(item)).as("round %s stage-two %s", round, item).isOne();
            }
          } else {
            AtomicIntegerArray visits = new AtomicIntegerArray(rows);
            executor.forEach(rows, visits::incrementAndGet);
            for (int row = 0; row < rows; row++) {
              assertThat(visits.get(row)).as("round %s row %s", round, row).isOne();
            }
          }
        }
      }
    } finally {
      GgufPersistentRowExecutor.setPollMillis(original);
    }
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void singleParticipantRunsEveryRowOnTheCaller(Implementation implementation) {
    try (GgufRowExecutor executor = implementation.factory().create(1, 1, "contract")) {
      Thread owner = Thread.currentThread();
      AtomicIntegerArray visits = new AtomicIntegerArray(257);
      IntConsumer operation =
          row -> {
            if (Thread.currentThread() != owner) {
              throw new AssertionError("a single-participant executor left its caller");
            }
            visits.incrementAndGet(row);
          };
      executor.forEach(257, operation);
      for (int row = 0; row < visits.length(); row++) {
        assertThat(visits.get(row)).as("row %s", row).isOne();
      }
    }
  }
}
