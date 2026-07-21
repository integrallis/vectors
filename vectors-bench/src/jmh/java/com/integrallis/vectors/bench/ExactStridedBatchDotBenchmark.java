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

import com.integrallis.vectors.core.VectorUtil;
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

/** Compares exact strided key-row batching with independent offset dot products. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 4, time = 1)
@Measurement(iterations = 5, time = 1)
public class ExactStridedBatchDotBenchmark {

  @Param({"64", "192", "512"})
  int rows;

  @Param({"128"})
  int columns;

  private int queryOffset;
  private int matrixOffset;
  private int rowStride;
  private int outOffset;
  private float[] query;
  private float[] matrix;
  private float[] out;

  @Setup(Level.Trial)
  public void setUp() {
    queryOffset = 7;
    matrixOffset = 11;
    rowStride = 1024;
    outOffset = 3;
    SplittableRandom random = new SplittableRandom(42L);
    query = randomFloats(random, queryOffset + columns + 5);
    matrix = randomFloats(random, matrixOffset + (rows - 1) * rowStride + columns + 5);
    out = new float[outOffset + rows + 5];
  }

  @Benchmark
  public void independentDots(Blackhole blackhole) {
    for (int row = 0; row < rows; row++) {
      out[outOffset + row] =
          VectorUtil.dotProduct(
              query, queryOffset, matrix, matrixOffset + row * rowStride, columns);
    }
    blackhole.consume(out);
  }

  @Benchmark
  public void exactTwoRowBatch(Blackhole blackhole) {
    VectorUtil.batchDotProductExact(
        query, queryOffset, matrix, matrixOffset, rowStride, rows, columns, out, outOffset);
    blackhole.consume(out);
  }

  private static float[] randomFloats(SplittableRandom random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = (float) random.nextDouble(-1.0, 1.0);
    }
    return values;
  }
}
