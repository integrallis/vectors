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

/** Compares GGUF GEMV with F32 activations against GGML-compatible Q8 activations. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufQuantizedMatVecBenchmark {

  @Param("1024")
  int rows;

  @Param("2048")
  int cols;

  private float[] query;
  private MemorySegment q4Weights;
  private MemorySegment q4KWeights;
  private MemorySegment q5Weights;
  private MemorySegment q8Weights;
  private MemorySegment q6Weights;
  private float[] out;
  private byte[] q8Quants;
  private float[] q8Scales;
  private short[] q8Sums;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(42L);
    query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    byte[] q4 = randomBlocks(random, rows * (cols / 32) * 18, 18, 0);
    byte[] q4K = randomQ4KBlocks(random, rows * (cols / 256) * 144);
    byte[] q5 = randomBlocks(random, rows * (cols / 32) * 22, 22, 0);
    byte[] q8 = randomBlocks(random, rows * (cols / 32) * 34, 34, 0);
    byte[] q6 = randomBlocks(random, rows * (cols / 256) * 210, 210, 208);
    q4Weights = MemorySegment.ofArray(q4);
    q4KWeights = MemorySegment.ofArray(q4K);
    q5Weights = MemorySegment.ofArray(q5);
    q8Weights = MemorySegment.ofArray(q8);
    q6Weights = MemorySegment.ofArray(q6);
    out = new float[rows];
    q8Quants = new byte[cols];
    q8Scales = new float[cols / 32];
    q8Sums = new short[cols / 16];
  }

  @Benchmark
  public void q4_0WithF32Activation(Blackhole blackhole) {
    VectorUtil.ggufQ4_0BatchDotProduct(query, q4Weights, rows, cols, out);
    blackhole.consume(out);
  }

  @Benchmark
  public void q4_0WithQ8_0Activation(Blackhole blackhole) {
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(query, q4Weights, rows, cols, out, q8Quants, q8Scales);
    blackhole.consume(out);
  }

  @Benchmark
  public void q4_KWithF32Activation(Blackhole blackhole) {
    VectorUtil.ggufQ4_KBatchDotProduct(query, q4KWeights, rows, cols, out);
    blackhole.consume(out);
  }

  @Benchmark
  public void q4_KWithQ8_KActivation(Blackhole blackhole) {
    VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
        query, q4KWeights, rows, cols, out, q8Quants, q8Scales, q8Sums);
    blackhole.consume(out);
  }

  @Benchmark
  public void q5_0WithF32Activation(Blackhole blackhole) {
    VectorUtil.ggufQ5_0BatchDotProduct(query, q5Weights, rows, cols, out);
    blackhole.consume(out);
  }

  @Benchmark
  public void q5_0WithQ8_0Activation(Blackhole blackhole) {
    VectorUtil.ggufQ5_0Q8_0BatchDotProduct(query, q5Weights, rows, cols, out, q8Quants, q8Scales);
    blackhole.consume(out);
  }

  @Benchmark
  public void q8_0WithF32Activation(Blackhole blackhole) {
    VectorUtil.ggufQ8_0BatchDotProduct(query, q8Weights, rows, cols, out);
    blackhole.consume(out);
  }

  @Benchmark
  public void q8_0WithQ8_0Activation(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchDotProduct(query, q8Weights, rows, cols, out, q8Quants, q8Scales);
    blackhole.consume(out);
  }

  @Benchmark
  public void q6_KWithF32Activation(Blackhole blackhole) {
    VectorUtil.ggufQ6_KBatchDotProduct(query, q6Weights, rows, cols, out);
    blackhole.consume(out);
  }

  @Benchmark
  public void q6_KWithQ8_KActivation(Blackhole blackhole) {
    VectorUtil.ggufQ6_KQ8_KBatchDotProduct(query, q6Weights, rows, cols, out, q8Quants, q8Scales);
    blackhole.consume(out);
  }

  private static byte[] randomBlocks(
      Random random, int byteCount, int blockBytes, int scaleOffset) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < byteCount; offset += blockBytes) {
      buffer.putShort(offset + scaleOffset, scale);
    }
    return blocks;
  }

  private static byte[] randomQ4KBlocks(Random random, int byteCount) {
    byte[] blocks = randomBlocks(random, byteCount, 144, 0);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short minScale = Float.floatToFloat16(0.005f);
    for (int offset = 0; offset < byteCount; offset += 144) {
      buffer.putShort(offset + Short.BYTES, minScale);
    }
    return blocks;
  }
}
