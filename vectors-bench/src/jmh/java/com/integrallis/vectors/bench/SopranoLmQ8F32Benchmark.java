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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.GgufQ8_0F32Matrix;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Measures the memory/throughput trade-off of expanding Soprano language-model Q8_0 projections to
 * an F32 execution layout. Expansion happens once during trial setup, outside the timed path.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 7, time = 1)
public class SopranoLmQ8F32Benchmark {

  private static final int BLOCK_SIZE = 32;
  private static final int Q8_BLOCK_BYTES = 34;

  @Param({"GATE_UP_2304x512", "DOWN_512x2304", "OUTPUT_HEAD_8192x512"})
  String shape;

  @Param({"1", "8", "16"})
  int batchSize;

  private int rows;
  private int columns;
  private MemorySegment q8Weights;
  private GgufQ8_0F32Matrix f32Weights;
  private float[] input;
  private float[] q8Output;
  private float[] f32Output;
  private byte[] q8ActivationQuants;
  private float[] q8ActivationScales;

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

    SplittableRandom random =
        new SplittableRandom(0x50_70_A0L + rows * 31L + columns * 17L + batchSize);
    input = new float[Math.multiplyExact(batchSize, columns)];
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    int q8Bytes = Math.multiplyExact(rows * (columns / BLOCK_SIZE), Q8_BLOCK_BYTES);
    q8Weights = MemorySegment.ofArray(randomQ8Blocks(random, q8Bytes));
    f32Weights = GgufQ8_0F32Matrix.from(q8Weights, rows, columns);
    q8Output = new float[Math.multiplyExact(batchSize, rows)];
    f32Output = new float[q8Output.length];

    // Reuse caller-owned production scratch on every invocation. Allocating this scratch inside
    // the benchmark would measure garbage creation rather than the Q8_0 matrix kernel.
    q8ActivationQuants = new byte[Math.multiplyExact(batchSize, columns)];
    q8ActivationScales = new float[Math.multiplyExact(batchSize, columns / BLOCK_SIZE)];
  }

  /** Production path: Q8_0 weights and per-call Q8_0 activation quantization. */
  @Benchmark
  public void productionQ8(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        input,
        q8Weights,
        batchSize,
        rows,
        columns,
        q8Output,
        q8ActivationQuants,
        q8ActivationScales);
    blackhole.consume(q8Output);
  }

  /** Candidate path: load-time-expanded F32 weights and F32 activations. */
  @Benchmark
  public void expandedF32(Blackhole blackhole) {
    f32Weights.multiplyBatch(input, batchSize, f32Output);
    blackhole.consume(f32Output);
  }

  public long serializedWeightBytes() {
    return f32Weights.serializedByteCount();
  }

  public long expandedWeightBytes() {
    return f32Weights.expandedByteCount();
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
