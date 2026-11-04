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
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * What one row-parallel dispatch costs, per executor implementation and thread count.
 *
 * <p>Decoding a token is a long sequence of short row-parallel dispatches, so the publication
 * mechanism is paid once per matrix per layer per token rather than amortised over a long parallel
 * region. {@code emptyDispatch} is the direct read of that cost: the row operation does almost
 * nothing, so the number is the publication and completion handshake and nothing else. The
 * projection benchmarks then show how much of it survives beside real arithmetic.
 *
 * <p>Swept over parallelism rather than fixed, because the interesting question is the <i>shape</i>
 * of the curve. A handshake whose cost grows with thread count takes back what the extra threads
 * earn, and the point where that crosses over is the only thing that decides a default. The sweep
 * deliberately runs past this machine's core count: an oversubscribed pool, where a spinning party
 * can hold the core another party needs to arrive on, is a real deployment state and the worst case
 * for any barrier.
 *
 * <p>Run both implementations over the whole sweep with:
 *
 * <pre>{@code
 * ./gradlew :vectors-bench:jmh -Pjmh.includes=GgufRowExecutorDispatchBenchmark
 * }</pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 2)
public class GgufRowExecutorDispatchBenchmark {

  /** Rows of a projection whose weights no longer fit a core-private cache. */
  private static final int PROJECTION_ROWS = 2048;

  private static final int PROJECTION_COLUMNS = 2048;

  /** Rows of a small dispatch, the size at which handshake cost rivals the work itself. */
  private static final int SMALL_ROWS = 128;

  @Param({"persistent", "countdown"})
  public String executorMode;

  @Param({"1", "2", "4", "6", "8", "12"})
  public int parallelism;

  private GgufRowExecutor executor;
  private float[] weights;
  private float[] input;
  private float[] output;
  private int[] touched;
  private IntConsumer emptyOperation;
  private IntConsumer smallOperation;
  private IntConsumer projectionOperation;

  @Setup(Level.Trial)
  public void setUp() {
    GgufParallelSupport.ExecutionMode mode = GgufParallelSupport.ExecutionMode.parse(executorMode);
    executor = GgufParallelSupport.newExecutor(mode, parallelism, 2, "dispatch-bench");

    weights = new float[PROJECTION_ROWS * PROJECTION_COLUMNS];
    input = new float[PROJECTION_COLUMNS];
    output = new float[PROJECTION_ROWS];
    touched = new int[PROJECTION_ROWS];
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (index % 251) * 0.00390625f;
    }
    for (int column = 0; column < input.length; column++) {
      input[column] = (column % 97) * 0.0078125f;
    }

    // Writes one int so the dispatch cannot be optimised away, and reads nothing shared, so the
    // measurement is the handshake rather than contention inside the row operation.
    emptyOperation = row -> touched[row]++;
    smallOperation = emptyOperation;
    projectionOperation = this::projectRow;
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    executor.close();
  }

  /** The handshake alone: as many rows as parties, each row almost free. */
  @Benchmark
  public void emptyDispatch(Blackhole blackhole) {
    executor.forEach(parallelism, emptyOperation);
    blackhole.consume(touched);
  }

  /** A dispatch too small to hide its own publication cost. */
  @Benchmark
  public void smallDispatch(Blackhole blackhole) {
    executor.forEach(SMALL_ROWS, smallOperation);
    blackhole.consume(touched);
  }

  /** A projection of the shape a decoder runs several times per layer. */
  @Benchmark
  public void projectionDispatch(Blackhole blackhole) {
    executor.forEach(PROJECTION_ROWS, projectionOperation);
    blackhole.consume(output);
  }

  private void projectRow(int row) {
    int base = row * PROJECTION_COLUMNS;
    float sum = 0.0f;
    for (int column = 0; column < PROJECTION_COLUMNS; column++) {
      sum += weights[base + column] * input[column];
    }
    output[row] = sum;
  }
}
