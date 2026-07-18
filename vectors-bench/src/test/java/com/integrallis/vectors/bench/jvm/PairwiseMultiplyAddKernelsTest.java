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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import org.junit.jupiter.api.Test;

class PairwiseMultiplyAddKernelsTest {

  private static final int TRIALS = 10_000;

  @Test
  void signedShortPairsProduceIntSums() {
    Random random = new Random(0x5eed5eedL);
    short[] left = new short[ShortVector.SPECIES_128.length()];
    short[] right = new short[ShortVector.SPECIES_128.length()];

    for (int trial = 0; trial < TRIALS; trial++) {
      for (int lane = 0; lane < left.length; lane++) {
        left[lane] = (short) random.nextInt();
        right[lane] = (short) random.nextInt();
      }

      int[] actual =
          PairwiseMultiplyAddKernels.multiplyAddSignedShorts(
                  ShortVector.fromArray(ShortVector.SPECIES_128, left, 0),
                  ShortVector.fromArray(ShortVector.SPECIES_128, right, 0))
              .toArray();

      for (int lane = 0; lane < actual.length; lane++) {
        int source = lane * 2;
        int expected = left[source] * right[source] + left[source + 1] * right[source + 1];
        assertThat(actual[lane]).isEqualTo(expected);
      }
    }
  }

  @Test
  void signedShortPairsProduceIntSumsAt256Bits() {
    Random random = new Random(0x2565eedL);
    short[] left = new short[ShortVector.SPECIES_256.length()];
    short[] right = new short[ShortVector.SPECIES_256.length()];

    for (int trial = 0; trial < TRIALS; trial++) {
      for (int lane = 0; lane < left.length; lane++) {
        left[lane] = (short) random.nextInt();
        right[lane] = (short) random.nextInt();
      }

      int[] actual =
          PairwiseMultiplyAddKernels.multiplyAddSignedShorts256(
                  ShortVector.fromArray(ShortVector.SPECIES_256, left, 0),
                  ShortVector.fromArray(ShortVector.SPECIES_256, right, 0))
              .toArray();

      for (int lane = 0; lane < actual.length; lane++) {
        int source = lane * 2;
        int expected = left[source] * right[source] + left[source + 1] * right[source + 1];
        assertThat(actual[lane]).isEqualTo(expected);
      }
    }
  }

  @Test
  void unsignedSignedBytePairsProduceSaturatedShortSums() {
    Random random = new Random(0x51a7eL);
    byte[] unsigned = new byte[ByteVector.SPECIES_128.length()];
    byte[] signed = new byte[ByteVector.SPECIES_128.length()];

    for (int trial = 0; trial < TRIALS; trial++) {
      random.nextBytes(unsigned);
      random.nextBytes(signed);

      short[] actual =
          PairwiseMultiplyAddKernels.multiplyAddUnsignedSignedBytesSaturating(
                  ByteVector.fromArray(ByteVector.SPECIES_128, unsigned, 0),
                  ByteVector.fromArray(ByteVector.SPECIES_128, signed, 0))
              .toArray();

      for (int lane = 0; lane < actual.length; lane++) {
        int source = lane * 2;
        int sum =
            Byte.toUnsignedInt(unsigned[source]) * signed[source]
                + Byte.toUnsignedInt(unsigned[source + 1]) * signed[source + 1];
        short expected = (short) Math.clamp(sum, Short.MIN_VALUE, Short.MAX_VALUE);
        assertThat(actual[lane]).isEqualTo(expected);
      }
    }
  }

  @Test
  void unsignedSignedBytePairsProduceSaturatedShortSumsAt256Bits() {
    Random random = new Random(0x25651a7eL);
    byte[] unsigned = new byte[ByteVector.SPECIES_256.length()];
    byte[] signed = new byte[ByteVector.SPECIES_256.length()];

    for (int trial = 0; trial < TRIALS; trial++) {
      random.nextBytes(unsigned);
      random.nextBytes(signed);

      short[] actual =
          PairwiseMultiplyAddKernels.multiplyAddUnsignedSignedBytesSaturating256(
                  ByteVector.fromArray(ByteVector.SPECIES_256, unsigned, 0),
                  ByteVector.fromArray(ByteVector.SPECIES_256, signed, 0))
              .toArray();

      for (int lane = 0; lane < actual.length; lane++) {
        int source = lane * 2;
        int sum =
            Byte.toUnsignedInt(unsigned[source]) * signed[source]
                + Byte.toUnsignedInt(unsigned[source + 1]) * signed[source + 1];
        short expected = (short) Math.clamp(sum, Short.MIN_VALUE, Short.MAX_VALUE);
        assertThat(actual[lane]).isEqualTo(expected);
      }
    }
  }
}
