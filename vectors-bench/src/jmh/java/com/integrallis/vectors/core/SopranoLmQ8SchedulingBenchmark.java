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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
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
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Separates Q8_0 scheduling from format effects for Soprano's batch-one LM projections.
 *
 * <p>This benchmark deliberately forces the production row-range kernel through the same persistent
 * executor used by production. It does not propose changing the Q8 parallel threshold.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 7, time = 1)
public class SopranoLmQ8SchedulingBenchmark {

  private static final int BATCH_SIZE = 1;
  private static final int BLOCK_SIZE = 32;
  private static final int Q8_BLOCK_BYTES = 34;
  private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();

  @Param({"GATE_UP_2304x512", "DOWN_512x2304", "OUTPUT_HEAD_8192x512"})
  String shape;

  private int rows;
  private int columns;
  private MemorySegment q8Weights;
  private float[] input;
  private float[] serialOutput;
  private float[] parallelOutput;
  private byte[] productionQuants;
  private float[] productionScales;
  private GgufQ8_0Batch activation;
  private GgufRowExecutor executor;

  @Setup(Level.Trial)
  public void setUp() {
    switch (shape) {
      case "GATE_UP_2304x512" -> {
        rows = 2304;
        columns = 512;
      }
      case "DOWN_512x2304" -> {
        rows = 512;
        columns = 2304;
      }
      case "OUTPUT_HEAD_8192x512" -> {
        rows = 8192;
        columns = 512;
      }
      default -> throw new IllegalArgumentException("unknown shape: " + shape);
    }

    SplittableRandom random = new SplittableRandom(0x50_71_A0L + rows * 31L + columns * 17L);
    input = new float[columns];
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    int q8Bytes = Math.multiplyExact(rows * (columns / BLOCK_SIZE), Q8_BLOCK_BYTES);
    q8Weights = MemorySegment.ofArray(randomQ8Blocks(random, q8Bytes));
    serialOutput = new float[rows];
    parallelOutput = new float[rows];
    productionQuants = new byte[columns];
    productionScales = new float[columns / BLOCK_SIZE];
    activation = GgufQ8_0Batch.allocate(BATCH_SIZE, columns);
    executor =
        GgufParallelSupport.newExecutor(
            GgufParallelSupport.ExecutionMode.PERSISTENT, PARALLELISM, 1, "soprano-lm-q8-bench");

    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        input,
        q8Weights,
        BATCH_SIZE,
        rows,
        columns,
        serialOutput,
        productionQuants,
        productionScales);
    forceParallelQ8();
    if (!Arrays.equals(serialOutput, parallelOutput)) {
      throw new IllegalStateException("forced row scheduling changed Q8_0 results");
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    executor.close();
  }

  /** Current production decision: this 1.18M-element operation remains serial. */
  @Benchmark
  public void productionSerialQ8(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        input,
        q8Weights,
        BATCH_SIZE,
        rows,
        columns,
        serialOutput,
        productionQuants,
        productionScales);
    blackhole.consume(serialOutput);
  }

  /** Same Q8_0 row kernel and quantization, forcibly divided into 12 persistent-worker chunks. */
  @Benchmark
  public void forcedRowParallelQ8(Blackhole blackhole) {
    forceParallelQ8();
    blackhole.consume(parallelOutput);
  }

  private void forceParallelQ8() {
    activation.quantize(input, BATCH_SIZE);
    executor.forEach(
        PARALLELISM,
        worker -> {
          int fromRow = worker * rows / PARALLELISM;
          int toRow = (worker + 1) * rows / PARALLELISM;
          if (fromRow < toRow) {
            VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
                q8Weights, BATCH_SIZE, rows, columns, fromRow, toRow, parallelOutput, activation);
          }
        });
  }

  private static byte[] randomQ8Blocks(SplittableRandom random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < blocks.length; offset += Q8_BLOCK_BYTES) {
      float scale = 0.001f + random.nextFloat() * 0.01f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }
    return blocks;
  }
}
