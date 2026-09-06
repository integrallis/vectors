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
package com.integrallis.vectors.bench.jvm;

import com.integrallis.vectors.core.GgufQ8_0Batch;
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

/** Evaluates true two-dimensional Q8 output tiling at the Soprano vocoder projection shape. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class Q8TiledMatmulBenchmark {

  private static final int BATCH_SIZE = 253;
  private static final int ROWS = 2304;
  private static final int COLS = 768;

  private MemorySegment weights;
  private byte[] activationQuants;
  private float[] activationScales;
  private float[] output;
  private GgufQ8_0Batch productionActivation;
  private Q8TiledMatmulKernels.Workspace workspace;

  @Setup(Level.Trial)
  public void setUp() {
    SplittableRandom random = new SplittableRandom(0x50f2a40L);
    float[] queries = new float[BATCH_SIZE * COLS];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    weights = MemorySegment.ofArray(randomQ8Blocks(random, ROWS * (COLS / 32) * 34));
    activationQuants = new byte[BATCH_SIZE * COLS];
    activationScales = new float[BATCH_SIZE * (COLS / 32)];
    output = new float[BATCH_SIZE * ROWS];
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, weights, BATCH_SIZE, ROWS, COLS, output, activationQuants, activationScales);
    productionActivation = GgufQ8_0Batch.allocate(BATCH_SIZE, COLS);
    productionActivation.quantize(queries, BATCH_SIZE);
    workspace = Q8TiledMatmulKernels.workspace(4, 8);
  }

  @Benchmark
  public void productionPrequantizedSingleThread(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(
        weights, BATCH_SIZE, ROWS, COLS, 0, ROWS, output, productionActivation);
    blackhole.consume(output);
  }

  @Benchmark
  public void tiledR1B1(Blackhole blackhole) {
    tiled(1, 1, blackhole);
  }

  @Benchmark
  public void tiledR1B2(Blackhole blackhole) {
    tiled(1, 2, blackhole);
  }

  @Benchmark
  public void tiledR1B4(Blackhole blackhole) {
    tiled(1, 4, blackhole);
  }

  @Benchmark
  public void tiledR1B8(Blackhole blackhole) {
    tiled(1, 8, blackhole);
  }

  @Benchmark
  public void tiledR2B1(Blackhole blackhole) {
    tiled(2, 1, blackhole);
  }

  @Benchmark
  public void tiledR2B2(Blackhole blackhole) {
    tiled(2, 2, blackhole);
  }

  @Benchmark
  public void tiledR2B4(Blackhole blackhole) {
    tiled(2, 4, blackhole);
  }

  @Benchmark
  public void tiledR2B8(Blackhole blackhole) {
    tiled(2, 8, blackhole);
  }

  @Benchmark
  public void tiledR4B1(Blackhole blackhole) {
    tiled(4, 1, blackhole);
  }

  @Benchmark
  public void tiledR4B2(Blackhole blackhole) {
    tiled(4, 2, blackhole);
  }

  @Benchmark
  public void tiledR4B4(Blackhole blackhole) {
    tiled(4, 4, blackhole);
  }

  @Benchmark
  public void tiledR4B8(Blackhole blackhole) {
    tiled(4, 8, blackhole);
  }

  private void tiled(int rowTile, int batchTile, Blackhole blackhole) {
    Q8TiledMatmulKernels.tiled(
        weights,
        BATCH_SIZE,
        ROWS,
        COLS,
        activationQuants,
        activationScales,
        output,
        rowTile,
        batchTile,
        workspace);
    blackhole.consume(output);
  }

  private static byte[] randomQ8Blocks(SplittableRandom random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += 34) {
      buffer.putShort(offset, scale);
    }
    return blocks;
  }
}
