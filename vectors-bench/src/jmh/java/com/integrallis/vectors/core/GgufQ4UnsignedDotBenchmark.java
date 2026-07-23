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
import java.lang.foreign.ValueLayout;
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

/** Measures unsigned-nibble Q4 arithmetic with precomputed Q8 zero-point corrections. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufQ4UnsignedDotBenchmark {

  private static final int BLOCK_BYTES = 18;
  private static final int BLOCK_SIZE = 32;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Param({"1", "32", "64"})
  int blocks;

  private MemorySegment weights;
  private byte[] q8Quants;
  private float[] q8Scales;
  private int[] zeroPointCorrections;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(0x514750414952L);
    byte[] weightBytes = new byte[blocks * BLOCK_BYTES];
    q8Quants = new byte[blocks * BLOCK_SIZE];
    q8Scales = new float[blocks];
    zeroPointCorrections = new int[blocks * 8];
    for (int block = 0; block < blocks; block++) {
      int weightOffset = block * BLOCK_BYTES;
      short scale = Float.floatToFloat16(0.001f + random.nextFloat() * 0.05f);
      weightBytes[weightOffset] = (byte) scale;
      weightBytes[weightOffset + 1] = (byte) (scale >>> 8);
      for (int index = 0; index < 16; index++) {
        weightBytes[weightOffset + Short.BYTES + index] = (byte) random.nextInt();
      }
      for (int index = 0; index < BLOCK_SIZE; index++) {
        q8Quants[block * BLOCK_SIZE + index] = (byte) random.nextInt(-127, 128);
      }
      for (int group = 0; group < 8; group++) {
        int start = block * BLOCK_SIZE + group * 4;
        zeroPointCorrections[block * 8 + group] =
            8 * (q8Quants[start] + q8Quants[start + 1] + q8Quants[start + 2] + q8Quants[start + 3]);
      }
      q8Scales[block] = 0.001f + random.nextFloat() * 0.05f;
    }
    weights = MemorySegment.ofArray(weightBytes);

    float expected = scalarReference();
    float actual = unsignedBytePairwise();
    if (Float.floatToRawIntBits(expected) != Float.floatToRawIntBits(actual)) {
      throw new IllegalStateException("Q4 unsigned-byte benchmark kernel disagrees");
    }
  }

  @Benchmark
  public float unsignedBytePairwise() {
    return PanamaVectorUtilSupport.q4_0Q8_0UnsignedPairwiseRowDot(
        weights, 0, blocks, q8Quants, q8Scales, zeroPointCorrections);
  }

  private float scalarReference() {
    float[] accumulator = new float[4];
    for (int block = 0; block < blocks; block++) {
      long blockOffset = (long) block * BLOCK_BYTES;
      float scale = Float.float16ToFloat(weights.get(LE_SHORT, blockOffset)) * q8Scales[block];
      for (int group = 0; group < 4; group++) {
        int sum = 0;
        for (int half = 0; half < 2; half++) {
          for (int lane = 0; lane < 4; lane++) {
            int index = half * 16 + group * 4 + lane;
            int packed =
                Byte.toUnsignedInt(
                    weights.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + (index & 15)));
            int nibble = half == 0 ? packed & 0x0F : packed >>> 4;
            sum += (nibble - 8) * q8Quants[block * BLOCK_SIZE + index];
          }
        }
        accumulator[group] = Math.fma(scale, sum, accumulator[group]);
      }
    }
    return reduce(accumulator);
  }

  private static float reduce(float[] accumulator) {
    float even = accumulator[0] + accumulator[2];
    float odd = accumulator[1] + accumulator[3];
    return even + odd;
  }
}
