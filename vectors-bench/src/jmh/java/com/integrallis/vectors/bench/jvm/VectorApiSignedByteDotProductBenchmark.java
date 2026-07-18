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

import java.lang.foreign.MemorySegment;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
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

/** Compares two Vector API shapes for one 32-byte GGUF Q8 dot-product block. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(
    value = 3,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class VectorApiSignedByteDotProductBenchmark {

  private MemorySegment left;
  private byte[] right;

  @Setup(Level.Trial)
  public void setUp() {
    SplittableRandom random = new SplittableRandom(0x5eed5eedL);
    byte[] leftBytes = new byte[32];
    right = new byte[32];
    random.nextBytes(leftBytes);
    random.nextBytes(right);
    left = MemorySegment.ofArray(leftBytes);
  }

  @Benchmark
  public int currentB2I() {
    return currentB2IKernel(left, right);
  }

  @Benchmark
  public int pairwiseB2S() {
    return pairwiseB2SKernel(left, right);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static int currentB2IKernel(MemorySegment left, byte[] right) {
    return SignedByteDotProductKernels.dotB2I256(left, right);
  }

  @CompilerControl(CompilerControl.Mode.DONT_INLINE)
  static int pairwiseB2SKernel(MemorySegment left, byte[] right) {
    return SignedByteDotProductKernels.dotPairwise256(left, right);
  }
}
