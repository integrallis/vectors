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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
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

/** Compares widened and packed-pairwise Q4_0 by Q8_0 row kernels. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class GgufQ4PairwiseDotBenchmark {

  private static final int BLOCK_BYTES = 18;
  private static final int BLOCK_SIZE = 32;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Param({"1", "32", "64"})
  int blocks;

  private MemorySegment weights;
  private byte[] q8Quants;
  private float[] q8Scales;

  @Setup(Level.Trial)
  public void setUp() {
    Random random = new Random(0x514750414952L);
    byte[] weightBytes = new byte[blocks * BLOCK_BYTES];
    q8Quants = new byte[blocks * BLOCK_SIZE];
    q8Scales = new float[blocks];
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
      q8Scales[block] = 0.001f + random.nextFloat() * 0.05f;
    }
    weights = MemorySegment.ofArray(weightBytes);

    float widened = widened();
    float pairwise = pairwise();
    float offsetPairwise = offsetPairwise();
    float offsetPairwise128 = offsetPairwise128();
    if (Float.floatToRawIntBits(widened) != Float.floatToRawIntBits(pairwise)
        || Float.floatToRawIntBits(widened) != Float.floatToRawIntBits(offsetPairwise)
        || Float.floatToRawIntBits(widened) != Float.floatToRawIntBits(offsetPairwise128)) {
      throw new IllegalStateException("Q4 pairwise benchmark kernels disagree");
    }
  }

  @Benchmark
  public float widened() {
    return rowDot(false);
  }

  @Benchmark
  public float pairwise() {
    return rowDot(true);
  }

  @Benchmark
  public float offsetPairwise() {
    FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
    for (int block = 0; block < blocks; block++) {
      long blockOffset = (long) block * BLOCK_BYTES;
      float scale = Float.float16ToFloat(weights.get(LE_SHORT, blockOffset)) * q8Scales[block];
      IntVector integerLanes =
          PanamaVectorUtilSupport.q4_0Q8_0OffsetPairwiseIntegerLanes(
              weights, blockOffset + Short.BYTES, q8Quants, block * BLOCK_SIZE);
      FloatVector products =
          (FloatVector) integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
      accumulator =
          PanamaVectorUtilSupport.fma(
              products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
    }
    return accumulator.reduceLanes(VectorOperators.ADD);
  }

  @Benchmark
  public float offsetPairwise128() {
    FloatVector lowAccumulator = FloatVector.zero(FloatVector.SPECIES_128);
    FloatVector highAccumulator = FloatVector.zero(FloatVector.SPECIES_128);
    for (int block = 0; block < blocks; block++) {
      long blockOffset = (long) block * BLOCK_BYTES;
      int quantOffset = block * BLOCK_SIZE;
      float scale = Float.float16ToFloat(weights.get(LE_SHORT, blockOffset)) * q8Scales[block];
      IntVector lowLanes =
          PanamaVectorUtilSupport.q4_0Q8_0OffsetPairwise128IntegerLanes(
              weights, blockOffset + Short.BYTES, q8Quants, quantOffset, false);
      IntVector highLanes =
          PanamaVectorUtilSupport.q4_0Q8_0OffsetPairwise128IntegerLanes(
              weights, blockOffset + Short.BYTES, q8Quants, quantOffset + 16, true);
      FloatVector scaleVector = FloatVector.broadcast(FloatVector.SPECIES_128, scale);
      lowAccumulator =
          PanamaVectorUtilSupport.fma(
              (FloatVector) lowLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
              scaleVector,
              lowAccumulator);
      highAccumulator =
          PanamaVectorUtilSupport.fma(
              (FloatVector) highLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_128, 0),
              scaleVector,
              highAccumulator);
    }
    float even =
        (highAccumulator.lane(0) + lowAccumulator.lane(0))
            + (highAccumulator.lane(2) + lowAccumulator.lane(2));
    float odd =
        (highAccumulator.lane(1) + lowAccumulator.lane(1))
            + (highAccumulator.lane(3) + lowAccumulator.lane(3));
    return even + odd;
  }

  private float rowDot(boolean usePairwise) {
    FloatVector accumulator = FloatVector.zero(FloatVector.SPECIES_256);
    for (int block = 0; block < blocks; block++) {
      long blockOffset = (long) block * BLOCK_BYTES;
      float scale = Float.float16ToFloat(weights.get(LE_SHORT, blockOffset)) * q8Scales[block];
      IntVector integerLanes =
          usePairwise
              ? PanamaVectorUtilSupport.q4_0Q8_0PairwiseIntegerLanes(
                  weights, blockOffset + Short.BYTES, q8Quants, block * BLOCK_SIZE)
              : PanamaVectorUtilSupport.q4_0Q8_0IntegerLanes(
                  weights, blockOffset + Short.BYTES, q8Quants, block * BLOCK_SIZE);
      FloatVector products =
          (FloatVector) integerLanes.convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0);
      accumulator =
          PanamaVectorUtilSupport.fma(
              products, FloatVector.broadcast(FloatVector.SPECIES_256, scale), accumulator);
    }
    return accumulator.reduceLanes(VectorOperators.ADD);
  }
}
