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

import java.util.Random;
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

/** Measures repeated row-executor handoffs using a decode-sized matrix-vector workload. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufRowDispatchBenchmark {

  @Param({"COMMON", "DEDICATED", "PERSISTENT"})
  String executionMode;

  @Param({"1", "141"})
  int projections;

  @Param("1024")
  int rows;

  @Param("1024")
  int cols;

  @Param("2")
  int chunksPerThread;

  private GgufRowExecutor executor;
  private float[] query;
  private float[] weights;
  private float[] output;
  private IntConsumer rowOperation;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(42L);
    query = randomFloats(random, cols);
    weights = randomFloats(random, rows * cols);
    output = new float[rows];
    rowOperation = this::dotRow;
    int parallelism = Runtime.getRuntime().availableProcessors();
    executor =
        GgufParallelSupport.newExecutor(
            GgufParallelSupport.ExecutionMode.parse(executionMode),
            parallelism,
            chunksPerThread,
            "vectors-bench");
  }

  @Benchmark
  public void projectionSequence(Blackhole blackhole) {
    for (int projection = 0; projection < projections; projection++) {
      executor.forEach(rows, rowOperation);
    }
    blackhole.consume(output);
  }

  private void dotRow(int row) {
    int offset = row * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
      sum = Math.fma(weights[offset + col], query[col], sum);
    }
    output[row] = sum;
  }

  private static float[] randomFloats(Random random, int size) {
    float[] values = new float[size];
    for (int index = 0; index < size; index++) {
      values[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    return values;
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    executor.close();
  }
}
