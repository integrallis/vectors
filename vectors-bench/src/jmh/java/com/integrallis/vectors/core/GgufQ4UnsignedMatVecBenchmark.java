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
package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;
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

/** Production-shaped Q4_0 GEMV gate for the unsigned-nibble zero-point formulation. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 7, time = 1)
public class GgufQ4UnsignedMatVecBenchmark {

  private static final int BLOCK_BYTES = 18;
  private static final int BLOCK_SIZE = 32;

  @Param("1024")
  int rows;

  @Param("2048")
  int cols;

  private float[] query;
  private MemorySegment weights;
  private float[] baselineOut;
  private float[] candidateOut;
  private byte[] baselineQ8Quants;
  private float[] baselineQ8Scales;
  private int[] baselineZeroPointCorrections;
  private byte[] candidateQ8Quants;
  private float[] candidateQ8Scales;
  private int[] candidateZeroPointCorrections;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(0x51475a504f494e54L);
    query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    int blocks = cols / BLOCK_SIZE;
    byte[] weightBytes = new byte[rows * blocks * BLOCK_BYTES];
    for (int block = 0; block < rows * blocks; block++) {
      int offset = block * BLOCK_BYTES;
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      weightBytes[offset] = (byte) scale;
      weightBytes[offset + 1] = (byte) (scale >>> 8);
      for (int index = Short.BYTES; index < BLOCK_BYTES; index++) {
        weightBytes[offset + index] = (byte) random.nextInt();
      }
    }
    weights = MemorySegment.ofArray(weightBytes);

    baselineOut = new float[rows];
    candidateOut = new float[rows];
    baselineQ8Quants = new byte[cols];
    baselineQ8Scales = new float[blocks];
    baselineZeroPointCorrections = new int[blocks * 8];
    candidateQ8Quants = new byte[cols];
    candidateQ8Scales = new float[blocks];
    candidateZeroPointCorrections = new int[blocks * 8];

    runBaseline();
    runCandidate();
    for (int row = 0; row < rows; row++) {
      if (Float.floatToRawIntBits(baselineOut[row]) != Float.floatToRawIntBits(candidateOut[row])) {
        throw new IllegalStateException("Q4 unsigned GEMV disagrees at row " + row);
      }
    }
  }

  @Benchmark
  public void shortPairwiseControl(Blackhole blackhole) {
    runBaseline();
    blackhole.consume(checksum(baselineOut));
  }

  @Benchmark
  public void unsignedZeroPointCandidate(Blackhole blackhole) {
    runCandidate();
    blackhole.consume(checksum(candidateOut));
  }

  private void runBaseline() {
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        weights,
        rows,
        cols,
        baselineOut,
        baselineQ8Quants,
        baselineQ8Scales,
        baselineZeroPointCorrections,
        GgufQ4Kernel.SHORT_PAIRWISE);
  }

  private void runCandidate() {
    VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
        query,
        weights,
        rows,
        cols,
        candidateOut,
        candidateQ8Quants,
        candidateQ8Scales,
        candidateZeroPointCorrections,
        GgufQ4Kernel.UNSIGNED_PAIRWISE);
  }

  private static int checksum(float[] values) {
    int hash = 1;
    for (float value : values) {
      hash = 31 * hash + Float.floatToRawIntBits(value);
    }
    return hash;
  }
}
