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

import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
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

/** Compares mapped BF16 matvec with the existing expanded F32 kernel at Qwen 2.5 shapes. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class BFloat16MatrixBenchmark {

  /** Attention, MLP up, MLP down, and tied vocabulary projection shapes. */
  @Param({"896x896", "4864x896", "896x4864", "151936x896"})
  String shape;

  private BFloat16Matrix matrix;
  private Arena arena;
  private MemorySegment mappedExpanded;
  private float[] input;
  private float[] expanded;
  private float[] output;
  private int rows;
  private int columns;

  /**
   * Creates deterministic BF16 weights and verifies both paths agree within F32 accumulation error.
   */
  @Setup(Level.Trial)
  public void setUp() {
    String[] dimensions = shape.split("x", 2);
    rows = Integer.parseInt(dimensions[0]);
    columns = Integer.parseInt(dimensions[1]);
    int values = Math.multiplyExact(rows, columns);
    byte[] packed = new byte[Math.multiplyExact(values, Short.BYTES)];
    ByteBuffer bits = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
    byte[] expandedBytes = new byte[Math.multiplyExact(values, Float.BYTES)];
    ByteBuffer expandedBits = ByteBuffer.wrap(expandedBytes).order(ByteOrder.LITTLE_ENDIAN);
    expanded = new float[values];
    input = new float[columns];
    output = new float[rows];
    Random random = new Random(0xBF16L + rows * 31L + columns);
    for (int index = 0; index < values; index++) {
      float sample = (random.nextFloat() - 0.5f) * 0.125f;
      short bf16 = (short) (Float.floatToRawIntBits(sample) >>> Short.SIZE);
      bits.putShort(bf16);
      expanded[index] = Float.intBitsToFloat(Short.toUnsignedInt(bf16) << Short.SIZE);
      expandedBits.putFloat(expanded[index]);
    }
    for (int index = 0; index < columns; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    arena = Arena.ofShared();
    MemorySegment mappedPacked = arena.allocate(packed.length, Long.BYTES);
    mappedPacked.copyFrom(MemorySegment.ofArray(packed));
    mappedExpanded = arena.allocate(expandedBytes.length, Long.BYTES);
    mappedExpanded.copyFrom(MemorySegment.ofArray(expandedBytes));
    matrix = BFloat16Matrix.of(mappedPacked, rows, columns);

    float[] compactOutput = new float[rows];
    float[] expandedOutput = new float[rows];
    matrix.multiply(input, compactOutput);
    VectorUtil.batchDotProduct(input, expanded, rows, columns, expandedOutput);
    for (int row = 0; row < rows; row++) {
      float tolerance = 1.0e-4f * Math.max(1.0f, Math.abs(expandedOutput[row]));
      if (Math.abs(compactOutput[row] - expandedOutput[row]) > tolerance) {
        throw new IllegalStateException(
            "BF16 and expanded benchmark matrices disagree at row " + row);
      }
    }
  }

  /** Releases the native segments after each parameterized trial. */
  @TearDown(Level.Trial)
  public void tearDown() {
    arena.close();
  }

  /** Measures zero-copy BF16 matvec, including conversion to F32 in vector registers. */
  @Benchmark
  public void mappedBfloat16MatVec(Blackhole blackhole) {
    matrix.multiply(input, output);
    blackhole.consume(output);
  }

  /** Measures the existing SIMD F32 matvec over a matrix expanded to twice the storage. */
  @Benchmark
  public void expandedF32MatVec(Blackhole blackhole) {
    VectorUtil.batchDotProduct(input, expanded, rows, columns, output);
    blackhole.consume(output);
  }

  /** Measures the existing SIMD F32 matvec over an off-heap-compatible byte segment. */
  @Benchmark
  public void mappedF32MatVec(Blackhole blackhole) {
    VectorUtil.ggufF32BatchDotProduct(input, mappedExpanded, rows, columns, output);
    blackhole.consume(output);
  }
}
