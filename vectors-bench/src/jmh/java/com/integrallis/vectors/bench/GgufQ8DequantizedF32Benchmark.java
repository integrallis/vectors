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
 * Tests the Soprano vocoder hypothesis that expanding Q8_0 weights once can make its large
 * prefill-style batches faster. The decoded matrix is prepared during trial setup, outside the
 * measured inference paths.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufQ8DequantizedF32Benchmark {

  @Param("253")
  int batchSize;

  @Param("2304")
  int rows;

  @Param("768")
  int cols;

  private byte[] serializedWeights;
  private MemorySegment weightSegment;
  private GgufQ8_0F32Matrix decodedWeights;
  private float[] queries;
  private float[] q8Out;
  private float[] f32Out;
  private byte[] activationQuants;
  private float[] activationScales;

  @Setup(Level.Trial)
  public void setUp() {
    SplittableRandom random = new SplittableRandom(0x50_80_A0L);
    queries = new float[Math.multiplyExact(batchSize, cols)];
    for (int index = 0; index < queries.length; index++) {
      queries[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    serializedWeights = randomQ8Blocks(random, rows * (cols / 32) * 34);
    weightSegment = MemorySegment.ofArray(serializedWeights);
    decodedWeights = GgufQ8_0F32Matrix.from(weightSegment, rows, cols);
    q8Out = new float[Math.multiplyExact(batchSize, rows)];
    f32Out = new float[q8Out.length];
    activationQuants = new byte[Math.multiplyExact(batchSize, cols)];
    activationScales = new float[Math.multiplyExact(batchSize, cols / 32)];
  }

  /** Current production path: Q8_0 weights multiplied by Q8_0-quantized activations. */
  @Benchmark
  public void currentQ8(Blackhole blackhole) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        queries, weightSegment, batchSize, rows, cols, q8Out, activationQuants, activationScales);
    blackhole.consume(q8Out);
  }

  /** Candidate path: load-time-decoded F32 weights, parallelized over independent rows. */
  @Benchmark
  public void decodedF32RowLocalParallel(Blackhole blackhole) {
    decodedWeights.multiplyBatch(queries, batchSize, f32Out);
    blackhole.consume(f32Out);
  }

  /** One-time model-load cost; intentionally excluded from both inference benchmarks above. */
  @Benchmark
  public void decodeWeightsAtLoad(Blackhole blackhole) {
    blackhole.consume(GgufQ8_0F32Matrix.from(weightSegment, rows, cols));
  }

  public long serializedWeightBytes() {
    return decodedWeights.serializedByteCount();
  }

  public long decodedWeightBytes() {
    return decodedWeights.expandedByteCount();
  }

  private static byte[] randomQ8Blocks(SplittableRandom random, int byteCount) {
    byte[] blocks = new byte[byteCount];
    random.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < blocks.length; offset += 34) {
      float scale = 0.001f + random.nextFloat() * 0.01f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }
    return blocks;
  }
}
