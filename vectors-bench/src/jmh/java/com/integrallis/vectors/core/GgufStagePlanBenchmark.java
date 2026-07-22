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

import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Isolates publication and completion cost for two dependent GGUF work stages. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector", "-XX:ActiveProcessorCount=8"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 2)
public class GgufStagePlanBenchmark {

  private static final int FIRST_WORK_ITEMS = 152;
  private static final int SECOND_WORK_ITEMS = 1024;

  private GgufPersistentRowExecutor executor;
  private GgufStagePlan plan;
  private int[] firstOutput;
  private int[] secondOutput;
  private IntConsumer firstOperation;
  private IntConsumer secondOperation;

  @Setup(Level.Trial)
  public void setUp() {
    firstOutput = new int[FIRST_WORK_ITEMS];
    secondOutput = new int[SECOND_WORK_ITEMS];
    firstOperation = item -> firstOutput[item]++;
    secondOperation = item -> secondOutput[item] += firstOutput[item % FIRST_WORK_ITEMS];
    executor = new GgufPersistentRowExecutor(8, 2, "vectors-stage-bench");
    plan =
        GgufStagePlan.of(
            GgufStagePlan.stage(FIRST_WORK_ITEMS, this::runFirstRange),
            GgufStagePlan.stage(SECOND_WORK_ITEMS, this::runSecondRange));
  }

  @Benchmark
  public void separateDispatches(Blackhole blackhole) {
    executor.forEach(FIRST_WORK_ITEMS, firstOperation);
    executor.forEach(SECOND_WORK_ITEMS, secondOperation);
    blackhole.consume(secondOutput);
  }

  @Benchmark
  public void stagedDispatch(Blackhole blackhole) {
    executor.execute(plan);
    blackhole.consume(secondOutput);
  }

  private void runFirstRange(int fromInclusive, int toExclusive) {
    for (int item = fromInclusive; item < toExclusive; item++) {
      firstOperation.accept(item);
    }
  }

  private void runSecondRange(int fromInclusive, int toExclusive) {
    for (int item = fromInclusive; item < toExclusive; item++) {
      secondOperation.accept(item);
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    executor.close();
  }
}
