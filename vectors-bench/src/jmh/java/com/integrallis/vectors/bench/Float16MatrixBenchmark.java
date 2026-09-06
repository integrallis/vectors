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

import com.integrallis.vectors.core.Float16Matrix;
import com.integrallis.vectors.core.VectorUtil;
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
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares mapped F16 execution with expanding Safetensors weights to heap F32. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class Float16MatrixBenchmark {

  /** DeBERTa-xsmall attention and feed-forward projection shapes. */
  @Param({"384x384", "1536x384", "384x1536"})
  String shape;

  /** A single pair and representative short/long reranker batches. */
  @Param({"1", "32", "128"})
  int batchSize;

  private Float16Matrix matrix;
  private float[] input;
  private float[] packedOutput;
  private float[] expanded;
  private float[] expandedOutput;
  private int rows;
  private int columns;
  private Random random;

  /** Builds deterministic F16 weights and proves both benchmark arms use the same values. */
  @Setup(Level.Trial)
  public void setUp() {
    String[] dimensions = shape.split("x", 2);
    rows = Integer.parseInt(dimensions[0]);
    columns = Integer.parseInt(dimensions[1]);
    int values = Math.multiplyExact(rows, columns);
    byte[] packed = new byte[Math.multiplyExact(values, Short.BYTES)];
    ByteBuffer bits = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
    expanded = new float[values];
    input = new float[Math.multiplyExact(batchSize, columns)];
    packedOutput = new float[Math.multiplyExact(batchSize, rows)];
    expandedOutput = new float[packedOutput.length];
    random = new Random(0xF16L + rows * 31L + columns * 17L + batchSize);
    for (int index = 0; index < values; index++) {
      short f16 = Float.floatToFloat16((random.nextFloat() - 0.5f) * 0.125f);
      bits.putShort(f16);
      expanded[index] = Float.float16ToFloat(f16);
    }
    for (int index = 0; index < input.length; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }
    matrix = Float16Matrix.of(MemorySegment.ofArray(packed), rows, columns);

    matrix.multiplyBatch(input, 0, batchSize, packedOutput, 0);
    expandedF32();
    for (int index = 0; index < packedOutput.length; index++) {
      float tolerance = 1.0e-4f * Math.max(1.0f, Math.abs(expandedOutput[index]));
      if (Math.abs(packedOutput[index] - expandedOutput[index]) > tolerance) {
        throw new IllegalStateException("F16 and expanded F32 disagree at output " + index);
      }
    }
  }

  /** Measures zero-copy mapped F16, including binary16-to-F32 conversion. */
  @Benchmark
  public void mappedFloat16(Blackhole blackhole) {
    matrix.multiplyBatch(input, 0, batchSize, packedOutput, 0);
    blackhole.consume(packedOutput);
  }

  /** Measures the same values after paying twice the persistent weight memory for heap F32. */
  @Benchmark
  public void expandedFloat32(Blackhole blackhole) {
    expandedF32();
    blackhole.consume(expandedOutput);
  }

  private void expandedF32() {
    for (int batch = 0; batch < batchSize; batch++) {
      VectorUtil.batchDotProductExact(
          input,
          batch * columns,
          expanded,
          0,
          columns,
          rows,
          columns,
          expandedOutput,
          batch * rows);
    }
  }
}
