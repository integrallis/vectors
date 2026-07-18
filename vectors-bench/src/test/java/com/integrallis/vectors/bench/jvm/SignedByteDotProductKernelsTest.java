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

import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

class SignedByteDotProductKernelsTest {

  private static final int BLOCK_SIZE = 32;
  private static final int TRIALS = 10_000;

  @Test
  void currentAndPairwiseKernelsMatchScalarAcrossRandomSignedByteBlocks() {
    Random random = new Random(0x08d07L);
    byte[] left = new byte[BLOCK_SIZE];
    byte[] right = new byte[BLOCK_SIZE];

    for (int trial = 0; trial < TRIALS; trial++) {
      random.nextBytes(left);
      random.nextBytes(right);

      int expected = 0;
      for (int lane = 0; lane < BLOCK_SIZE; lane++) {
        expected += left[lane] * right[lane];
      }

      MemorySegment leftSegment = MemorySegment.ofArray(left);
      assertThat(SignedByteDotProductKernels.dotB2I256(leftSegment, right)).isEqualTo(expected);
      assertThat(SignedByteDotProductKernels.dotPairwise256(leftSegment, right))
          .isEqualTo(expected);
    }
  }
}
