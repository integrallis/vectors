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
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares repeated SAXPY calls with the four-row weighted accumulation kernel. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 4, time = 1)
@Measurement(iterations = 5, time = 1)
public class WeightedRowsBenchmark {

  @Param({"64", "192", "512"})
  int rows;

  @Param({"128"})
  int columns;

  private int rowStride;
  private float[] matrix;
  private float[] weights;
  private float[] out;

  @Setup(Level.Trial)
  public void setUp() {
    rowStride = 1024;
    SplittableRandom random = new SplittableRandom(42L);
    matrix = new float[rows * rowStride];
    weights = new float[rows];
    out = new float[columns];
    for (int index = 0; index < matrix.length; index++) {
      matrix[index] = (float) random.nextDouble(-1.0, 1.0);
    }
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) random.nextDouble(-1.0, 1.0);
    }
  }

  @Benchmark
  public void repeatedAddScaled(Blackhole blackhole) {
    Arrays.fill(out, 0.0f);
    for (int row = 0; row < rows; row++) {
      VectorUtil.addScaledInPlace(out, 0, matrix, row * rowStride, columns, weights[row]);
    }
    blackhole.consume(out);
  }

  @Benchmark
  public void fourRowWeightedSum(Blackhole blackhole) {
    Arrays.fill(out, 0.0f);
    VectorUtil.addWeightedRowsInPlace(out, 0, matrix, 0, rowStride, weights, 0, rows, columns);
    blackhole.consume(out);
  }
}
