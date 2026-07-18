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

import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.CompilerControl;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Standalone reproducer for packed pairwise multiply-add lowering.
 *
 * <p>Run with {@code ./gradlew :vectors-bench:jmh
 * -Pjmh.includes=VectorApiPairwiseMultiplyAddBenchmark}. Compare a stable JDK 25 runtime with a
 * post-{@code 0bc546878361} Graal development build and inspect generated code for {@code
 * VPMADDWD}/{@code VPMADDUBSW}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class VectorApiPairwiseMultiplyAddBenchmark {

  private ShortVector signedShorts;
  private ShortVector signedShortWeights;
  private ShortVector signedShorts256;
  private ShortVector signedShortWeights256;
  private ByteVector unsignedBytes;
  private ByteVector signedByteWeights;
  private ByteVector unsignedBytes256;
  private ByteVector signedByteWeights256;

  @Setup(Level.Trial)
  public void setUp() {
    SplittableRandom random = new SplittableRandom(0x5eed5eedL);

    short[] shorts = new short[ShortVector.SPECIES_128.length()];
    short[] shortWeights = new short[ShortVector.SPECIES_128.length()];
    for (int lane = 0; lane < shorts.length; lane++) {
      shorts[lane] = (short) random.nextInt();
      shortWeights[lane] = (short) random.nextInt();
    }
    signedShorts = ShortVector.fromArray(ShortVector.SPECIES_128, shorts, 0);
    signedShortWeights = ShortVector.fromArray(ShortVector.SPECIES_128, shortWeights, 0);

    short[] shorts256 = new short[ShortVector.SPECIES_256.length()];
    short[] shortWeights256 = new short[ShortVector.SPECIES_256.length()];
    for (int lane = 0; lane < shorts256.length; lane++) {
      shorts256[lane] = (short) random.nextInt();
      shortWeights256[lane] = (short) random.nextInt();
    }
    signedShorts256 = ShortVector.fromArray(ShortVector.SPECIES_256, shorts256, 0);
    signedShortWeights256 = ShortVector.fromArray(ShortVector.SPECIES_256, shortWeights256, 0);

    byte[] bytes = new byte[ByteVector.SPECIES_128.length()];
    byte[] byteWeights = new byte[ByteVector.SPECIES_128.length()];
    random.nextBytes(bytes);
    random.nextBytes(byteWeights);
    unsignedBytes = ByteVector.fromArray(ByteVector.SPECIES_128, bytes, 0);
    signedByteWeights = ByteVector.fromArray(ByteVector.SPECIES_128, byteWeights, 0);

    byte[] bytes256 = new byte[ByteVector.SPECIES_256.length()];
    byte[] byteWeights256 = new byte[ByteVector.SPECIES_256.length()];
    random.nextBytes(bytes256);
    random.nextBytes(byteWeights256);
    unsignedBytes256 = ByteVector.fromArray(ByteVector.SPECIES_256, bytes256, 0);
    signedByteWeights256 = ByteVector.fromArray(ByteVector.SPECIES_256, byteWeights256, 0);
  }

  @Benchmark
  public int signedShortPairwiseMultiplyAdd() {
    return signedShortPairwiseMultiplyAddKernel(signedShorts, signedShortWeights);
  }

  @Benchmark
  public short unsignedSignedBytePairwiseMultiplyAdd() {
    return unsignedSignedBytePairwiseMultiplyAddKernel(unsignedBytes, signedByteWeights);
  }

  @Benchmark
  public int signedShortPairwiseMultiplyAdd256() {
    return signedShortPairwiseMultiplyAddKernel256(signedShorts256, signedShortWeights256);
  }

  @Benchmark
  public short unsignedSignedBytePairwiseMultiplyAdd256() {
    return unsignedSignedBytePairwiseMultiplyAddKernel256(unsignedBytes256, signedByteWeights256);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static int signedShortPairwiseMultiplyAddKernel(ShortVector left, ShortVector right) {
    return PairwiseMultiplyAddKernels.multiplyAddSignedShorts(left, right)
        .reduceLanes(VectorOperators.ADD);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static short unsignedSignedBytePairwiseMultiplyAddKernel(ByteVector left, ByteVector right) {
    return PairwiseMultiplyAddKernels.multiplyAddUnsignedSignedBytesSaturating(left, right)
        .reduceLanes(VectorOperators.ADD);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static int signedShortPairwiseMultiplyAddKernel256(ShortVector left, ShortVector right) {
    return PairwiseMultiplyAddKernels.multiplyAddSignedShorts256(left, right)
        .reduceLanes(VectorOperators.ADD);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static short unsignedSignedBytePairwiseMultiplyAddKernel256(ByteVector left, ByteVector right) {
    return PairwiseMultiplyAddKernels.multiplyAddUnsignedSignedBytesSaturating256(left, right)
        .reduceLanes(VectorOperators.ADD);
  }
}
