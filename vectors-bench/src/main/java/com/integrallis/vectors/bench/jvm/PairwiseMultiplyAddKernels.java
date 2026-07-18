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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

/** Standalone Vector API graphs used to verify pairwise multiply-add compiler lowering. */
public final class PairwiseMultiplyAddKernels {

  private static final VectorShuffle<Short> SHORT_EVEN_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_128, 0);
  private static final VectorShuffle<Short> SHORT_ODD_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_128, 1);
  private static final VectorShuffle<Short> SHORT_256_EVEN_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_256, 0);
  private static final VectorShuffle<Short> SHORT_256_ODD_LANES =
      VectorShuffle.makeUnzip(ShortVector.SPECIES_256, 1);
  private static final VectorShuffle<Byte> BYTE_EVEN_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 0);
  private static final VectorShuffle<Byte> BYTE_ODD_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 1);
  private static final VectorShuffle<Byte> BYTE_256_EVEN_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_256, 0);
  private static final VectorShuffle<Byte> BYTE_256_ODD_LANES =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_256, 1);

  private PairwiseMultiplyAddKernels() {}

  /** Equivalent to the x86 {@code PMADDWD} operation for a 128-bit vector. */
  public static IntVector multiplyAddSignedShorts(ShortVector left, ShortVector right) {
    ShortVector leftEven = left.rearrange(SHORT_EVEN_LANES);
    ShortVector leftOdd = left.rearrange(SHORT_ODD_LANES);
    ShortVector rightEven = right.rearrange(SHORT_EVEN_LANES);
    ShortVector rightOdd = right.rearrange(SHORT_ODD_LANES);

    IntVector leftEvenInts =
        (IntVector) leftEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector leftOddInts =
        (IntVector) leftOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector rightEvenInts =
        (IntVector) rightEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
    IntVector rightOddInts =
        (IntVector) rightOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);

    return leftEvenInts.mul(rightEvenInts).add(leftOddInts.mul(rightOddInts));
  }

  /** Equivalent to the x86 {@code VPMADDWD} operation for a 256-bit vector. */
  public static IntVector multiplyAddSignedShorts256(ShortVector left, ShortVector right) {
    ShortVector leftEven = left.rearrange(SHORT_256_EVEN_LANES);
    ShortVector leftOdd = left.rearrange(SHORT_256_ODD_LANES);
    ShortVector rightEven = right.rearrange(SHORT_256_EVEN_LANES);
    ShortVector rightOdd = right.rearrange(SHORT_256_ODD_LANES);

    IntVector leftEvenInts =
        (IntVector) leftEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector leftOddInts =
        (IntVector) leftOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector rightEvenInts =
        (IntVector) rightEven.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);
    IntVector rightOddInts =
        (IntVector) rightOdd.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0);

    return leftEvenInts.mul(rightEvenInts).add(leftOddInts.mul(rightOddInts));
  }

  /** Equivalent to the x86 {@code PMADDUBSW} operation for a 128-bit vector. */
  public static ShortVector multiplyAddUnsignedSignedBytesSaturating(
      ByteVector unsigned, ByteVector signed) {
    ByteVector unsignedEven = unsigned.rearrange(BYTE_EVEN_LANES);
    ByteVector unsignedOdd = unsigned.rearrange(BYTE_ODD_LANES);
    ByteVector signedEven = signed.rearrange(BYTE_EVEN_LANES);
    ByteVector signedOdd = signed.rearrange(BYTE_ODD_LANES);

    ShortVector unsignedEvenShorts =
        (ShortVector)
            unsignedEven.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_128, 0);
    ShortVector unsignedOddShorts =
        (ShortVector)
            unsignedOdd.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_128, 0);
    ShortVector signedEvenShorts =
        (ShortVector) signedEven.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
    ShortVector signedOddShorts =
        (ShortVector) signedOdd.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);

    return unsignedEvenShorts
        .mul(signedEvenShorts)
        .lanewise(VectorOperators.SADD, unsignedOddShorts.mul(signedOddShorts));
  }

  /** Equivalent to the x86 {@code VPMADDUBSW} operation for a 256-bit vector. */
  public static ShortVector multiplyAddUnsignedSignedBytesSaturating256(
      ByteVector unsigned, ByteVector signed) {
    ByteVector unsignedEven = unsigned.rearrange(BYTE_256_EVEN_LANES);
    ByteVector unsignedOdd = unsigned.rearrange(BYTE_256_ODD_LANES);
    ByteVector signedEven = signed.rearrange(BYTE_256_EVEN_LANES);
    ByteVector signedOdd = signed.rearrange(BYTE_256_ODD_LANES);

    ShortVector unsignedEvenShorts =
        (ShortVector)
            unsignedEven.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_256, 0);
    ShortVector unsignedOddShorts =
        (ShortVector)
            unsignedOdd.convertShape(VectorOperators.ZERO_EXTEND_B2S, ShortVector.SPECIES_256, 0);
    ShortVector signedEvenShorts =
        (ShortVector) signedEven.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
    ShortVector signedOddShorts =
        (ShortVector) signedOdd.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

    return unsignedEvenShorts
        .mul(signedEvenShorts)
        .lanewise(VectorOperators.SADD, unsignedOddShorts.mul(signedOddShorts));
  }
}
