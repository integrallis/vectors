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

import com.integrallis.vectors.core.GgufQ4Kernel;
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

/** Compares weight-reusing Q4_0 batched matmul with independent GEMV calls. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufBatchedMatmulBenchmark {

  @Param({"1", "2", "4", "8", "24", "32"})
  int batchSize;

  @Param("1024")
  int rows;

  @Param("2048")
  int cols;

  private float[] queries;
  private float[][] independentQueries;
  private MemorySegment weights;
  private float[] batchedOut;
  private float[] independentOut;
  private byte[] batchedQuants;
  private float[] batchedScales;
  private int[] batchedZeroPointCorrections;
  private float[] batchedLanes;
  private byte[] independentQuants;
  private float[] independentScales;
  private int[] independentZeroPointCorrections;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(42L);
    queries = new float[batchSize * cols];
    independentQueries = new float[batchSize][cols];
    for (int batch = 0; batch < batchSize; batch++) {
      for (int col = 0; col < cols; col++) {
        float value = random.nextFloat() * 2.0f - 1.0f;
        queries[batch * cols + col] = value;
        independentQueries[batch][col] = value;
      }
    }

    byte[] blocks = new byte[rows * (cols / 32) * 18];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += 18) {
      buffer.putShort(offset, scale);
    }
    weights = MemorySegment.ofArray(blocks);
    batchedOut = new float[batchSize * rows];
    independentOut = new float[rows];
    batchedQuants = new byte[batchSize * cols];
    batchedScales = new float[batchSize * (cols / 32)];
    batchedZeroPointCorrections = new int[batchSize * cols / 4];
    batchedLanes = new float[batchSize * rows * 8];
    independentQuants = new byte[cols];
    independentScales = new float[cols / 32];
    independentZeroPointCorrections = new int[cols / 4];
  }

  @Benchmark
  public void batched(Blackhole blackhole) {
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        batchedOut,
        batchedQuants,
        batchedScales,
        batchedZeroPointCorrections,
        batchedLanes);
    blackhole.consume(batchedOut);
  }

  @Benchmark
  public void independent(Blackhole blackhole) {
    for (float[] query : independentQueries) {
      VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
          query,
          weights,
          rows,
          cols,
          independentOut,
          independentQuants,
          independentScales,
          independentZeroPointCorrections);
      blackhole.consume(independentOut);
    }
  }

  @Benchmark
  public void unsignedBatched(Blackhole blackhole) {
    VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        batchedOut,
        batchedQuants,
        batchedScales,
        batchedZeroPointCorrections,
        batchedLanes,
        GgufQ4Kernel.UNSIGNED_PAIRWISE);
    blackhole.consume(batchedOut);
  }

  @Benchmark
  public void unsignedIndependent(Blackhole blackhole) {
    for (float[] query : independentQueries) {
      VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
          query,
          weights,
          rows,
          cols,
          independentOut,
          independentQuants,
          independentScales,
          independentZeroPointCorrections,
          GgufQ4Kernel.UNSIGNED_PAIRWISE);
      blackhole.consume(independentOut);
    }
  }
}
