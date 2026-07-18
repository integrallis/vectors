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
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

/** Standalone 32-byte signed dot products matching one GGUF Q8 block. */
public final class SignedByteDotProductKernels {

  private static final int BLOCK_SIZE = 32;

  private SignedByteDotProductKernels() {}

  /** Current JDK 25-compatible shape: widen eight byte products directly to int lanes. */
  public static int dotB2I256(MemorySegment left, byte[] right) {
    IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
    for (int index = 0; index < BLOCK_SIZE; index += ByteVector.SPECIES_64.length()) {
      ByteVector leftBytes =
          ByteVector.fromMemorySegment(ByteVector.SPECIES_64, left, index, ByteOrder.LITTLE_ENDIAN);
      ByteVector rightBytes = ByteVector.fromArray(ByteVector.SPECIES_64, right, index);
      IntVector leftInts =
          (IntVector) leftBytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      IntVector rightInts =
          (IntVector) rightBytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
      accumulator = accumulator.add(leftInts.mul(rightInts));
    }
    return accumulator.reduceLanes(VectorOperators.ADD);
  }

  /** Graal pairwise shape: widen sixteen bytes to shorts, then lower pairs to {@code VPMADDWD}. */
  public static int dotPairwise256(MemorySegment left, byte[] right) {
    IntVector accumulator = IntVector.zero(IntVector.SPECIES_256);
    for (int index = 0; index < BLOCK_SIZE; index += ByteVector.SPECIES_128.length()) {
      ShortVector leftShorts =
          (ShortVector)
              ByteVector.fromMemorySegment(
                      ByteVector.SPECIES_128, left, index, ByteOrder.LITTLE_ENDIAN)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      ShortVector rightShorts =
          (ShortVector)
              ByteVector.fromArray(ByteVector.SPECIES_128, right, index)
                  .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
      accumulator =
          accumulator.add(
              PairwiseMultiplyAddKernels.multiplyAddSignedShorts256(leftShorts, rightShorts));
    }
    return accumulator.reduceLanes(VectorOperators.ADD);
  }
}
