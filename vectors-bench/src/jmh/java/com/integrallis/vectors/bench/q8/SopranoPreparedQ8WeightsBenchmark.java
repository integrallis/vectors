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
package com.integrallis.vectors.bench.q8;

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
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Tests whether one-time Q8_0 weight preparation helps Soprano's largest vocoder projection. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(
    value = 1,
    jvmArgsPrepend = {
      "--add-modules",
      "jdk.incubator.vector",
      "-Djava.util.concurrent.ForkJoinPool.common.parallelism=12"
    })
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class SopranoPreparedQ8WeightsBenchmark {

  private static final int BATCH_SIZE = 253;
  private static final int ROWS = 2304;
  private static final int COLS = 768;
  private static final int BLOCK_SIZE = 32;
  private static final int Q8_BLOCK_BYTES = 34;

  private float[] queries;
  private MemorySegment interleavedWeights;
  private PreparedQ8Weights preparedWeights;
  private float[] baselineOut;
  private float[] preparedOut;
  private byte[] baselineQuants;
  private float[] baselineScales;
  private byte[] preparedQuants;
  private float[] preparedScales;

  @Setup(Level.Trial)
  public void setUp() {
    SplittableRandom random = new SplittableRandom(0x50A12A0L);
    queries = new float[BATCH_SIZE * COLS];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    byte[] blocks = randomQ8Blocks(random, ROWS * (COLS / BLOCK_SIZE) * Q8_BLOCK_BYTES);
    interleavedWeights = MemorySegment.ofArray(blocks);
    preparedWeights = PreparedQ8Weights.from(interleavedWeights, ROWS, COLS);
    baselineOut = new float[BATCH_SIZE * ROWS];
    preparedOut = new float[BATCH_SIZE * ROWS];
    baselineQuants = new byte[BATCH_SIZE * COLS];
    baselineScales = new float[BATCH_SIZE * (COLS / BLOCK_SIZE)];
    preparedQuants = new byte[BATCH_SIZE * COLS];
    preparedScales = new float[BATCH_SIZE * (COLS / BLOCK_SIZE)];
  }

  @Benchmark
  public void interleavedVectorUtil(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries,
        interleavedWeights,
        BATCH_SIZE,
        ROWS,
        COLS,
        baselineOut,
        baselineQuants,
        baselineScales);
    blackhole.consume(baselineOut);
  }

  @Benchmark
  public void preparedSplitArrays(Blackhole blackhole) {
    preparedWeights.multiply(queries, BATCH_SIZE, preparedOut, preparedQuants, preparedScales);
    blackhole.consume(preparedOut);
  }

  private static byte[] randomQ8Blocks(SplittableRandom random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += Q8_BLOCK_BYTES) {
      buffer.putShort(offset, scale);
    }
    return blocks;
  }
}
