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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
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

/** Java MXFP4 fused-decode GEMV compared with the same weights materialized as F32. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 7, time = 1)
public class Mxfp4MatVecBenchmark {

  @Param("1024")
  int rows;

  @Param("2048")
  int columns;

  @Param({"heap", "native"})
  String storage;

  private float[] input;
  private float[] materializedWeights;
  private float[] materializedOutput;
  private float[] packedOutput;
  private Mxfp4Matrix packedMatrix;
  private GgufQ8_0Batch q8Activation;

  @Setup(Level.Trial)
  public void setUp() {
    RandomGenerator random =
        RandomGeneratorFactory.<RandomGenerator>of("Random").create(0x4d58465034L);
    input = new float[columns];
    for (int index = 0; index < columns; index++) {
      input[index] = random.nextFloat() * 2.0f - 1.0f;
    }

    byte[] blocks = new byte[Math.multiplyExact(rows, columns / 2)];
    random.nextBytes(blocks);
    byte[] scales = new byte[Math.multiplyExact(rows, columns / 32)];
    for (int index = 0; index < scales.length; index++) {
      scales[index] = (byte) (124 + random.nextInt(7));
    }
    MemorySegment blockStorage = MemorySegment.ofArray(blocks);
    MemorySegment scaleStorage = MemorySegment.ofArray(scales);
    if ("native".equals(storage)) {
      Arena arena = Arena.ofAuto();
      blockStorage = arena.allocate(blocks.length);
      scaleStorage = arena.allocate(scales.length);
      MemorySegment.copy(blocks, 0, blockStorage, ValueLayout.JAVA_BYTE, 0, blocks.length);
      MemorySegment.copy(scales, 0, scaleStorage, ValueLayout.JAVA_BYTE, 0, scales.length);
    }
    packedMatrix = Mxfp4Matrix.of(blockStorage, scaleStorage, rows, columns);
    q8Activation = GgufQ8_0Batch.allocate(1, columns);

    materializedWeights = new float[Math.multiplyExact(rows, columns)];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        materializedWeights[row * columns + column] = packedMatrix.value(row, column);
      }
    }
    materializedOutput = new float[rows];
    packedOutput = new float[rows];
    runMaterialized();
    runPacked();
    for (int row = 0; row < rows; row++) {
      float tolerance = 0.001f * (1.0f + Math.abs(materializedOutput[row]));
      if (Math.abs(materializedOutput[row] - packedOutput[row]) > tolerance) {
        throw new IllegalStateException(
            "MXFP4 GEMV disagrees at row "
                + row
                + ": "
                + materializedOutput[row]
                + " != "
                + packedOutput[row]);
      }
    }
    runQ8();
    if (cosine(materializedOutput, packedOutput) < 0.9999) {
      throw new IllegalStateException("MXFP4 W4A8 result fell below the cosine gate");
    }
  }

  @Benchmark
  public void materializedF32Control(Blackhole blackhole) {
    runMaterialized();
    blackhole.consume(checksum(materializedOutput));
  }

  @Benchmark
  public void packedMxfp4Candidate(Blackhole blackhole) {
    runPacked();
    blackhole.consume(checksum(packedOutput));
  }

  @Benchmark
  public void q8ActivationMxfp4Candidate(Blackhole blackhole) {
    runQ8();
    blackhole.consume(checksum(packedOutput));
  }

  @Benchmark
  public void preparedQ8ActivationMxfp4Candidate(Blackhole blackhole) {
    runPreparedQ8();
    blackhole.consume(checksum(packedOutput));
  }

  private void runMaterialized() {
    VectorUtil.batchDotProduct(input, materializedWeights, rows, columns, materializedOutput);
  }

  private void runPacked() {
    packedMatrix.multiply(input, packedOutput);
  }

  private void runQ8() {
    packedMatrix.multiplyQ8(input, packedOutput, q8Activation);
  }

  private void runPreparedQ8() {
    packedMatrix.multiplyQ8(q8Activation, packedOutput);
  }

  private static double cosine(float[] first, float[] second) {
    double dot = 0.0;
    double firstNorm = 0.0;
    double secondNorm = 0.0;
    for (int index = 0; index < first.length; index++) {
      dot += (double) first[index] * second[index];
      firstNorm += (double) first[index] * first[index];
      secondNorm += (double) second[index] * second[index];
    }
    return dot / Math.sqrt(firstNorm * secondNorm);
  }

  private static int checksum(float[] values) {
    int hash = 1;
    for (float value : values) {
      hash = 31 * hash + Float.floatToRawIntBits(value);
    }
    return hash;
  }
}
