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

/** Compares row-local Q4_K batched matmul with independent GEMV calls. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufQ4KBatchedMatmulBenchmark {

  @Param({"1", "2", "4", "8", "32"})
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
  private short[] batchedSums;
  private byte[] independentQuants;
  private float[] independentScales;
  private short[] independentSums;

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

    byte[] blocks = randomQ4KBlocks(random, rows * (cols / 256) * 144);
    weights = MemorySegment.ofArray(blocks);
    batchedOut = new float[batchSize * rows];
    independentOut = new float[rows];
    batchedQuants = new byte[batchSize * cols];
    batchedScales = new float[batchSize * (cols / 256)];
    batchedSums = new short[batchSize * (cols / 16)];
    independentQuants = new byte[cols];
    independentScales = new float[cols / 256];
    independentSums = new short[cols / 16];
  }

  private static byte[] randomQ4KBlocks(Random random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    short minScale = Float.floatToFloat16(0.005f);
    for (int offset = 0; offset < blocks.length; offset += 144) {
      buffer.putShort(offset, scale);
      buffer.putShort(offset + Short.BYTES, minScale);
    }
    return blocks;
  }

  @Benchmark
  public void batched(Blackhole blackhole) {
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        weights,
        batchSize,
        rows,
        cols,
        batchedOut,
        batchedQuants,
        batchedScales,
        batchedSums);
    blackhole.consume(batchedOut);
  }

  @Benchmark
  public void independent(Blackhole blackhole) {
    for (float[] query : independentQueries) {
      VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
          query,
          weights,
          rows,
          cols,
          independentOut,
          independentQuants,
          independentScales,
          independentSums);
      blackhole.consume(independentOut);
    }
  }
}
