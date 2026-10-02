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
package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.util.SplittableRandom;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

/** The combined CRC must equal what {@link CRC32} computes over the concatenation. */
class Crc32CombineTest {

  @Test
  void combineMatchesTheReferenceOverRandomSplits() {
    SplittableRandom random = new SplittableRandom(4242);
    for (int trial = 0; trial < 2_000; trial++) {
      int lenA = random.nextInt(0, 4_000);
      int lenB = random.nextInt(0, 4_000);
      byte[] a = bytes(random, lenA);
      byte[] b = bytes(random, lenB);

      assertThat(Crc32Combine.combine(crc(a), crc(b), b.length))
          .as("lenA=%d lenB=%d", lenA, lenB)
          .isEqualTo(crc(concat(a, b)));
    }
  }

  @Test
  void combineHandlesEdgeLengths() {
    SplittableRandom random = new SplittableRandom(7);
    byte[] a = bytes(random, 1_000);
    for (int lenB :
        new int[] {0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 257, 4095, 4096}) {
      byte[] b = bytes(random, lenB);
      assertThat(Crc32Combine.combine(crc(a), crc(b), lenB))
          .as("lenB=%d", lenB)
          .isEqualTo(crc(concat(a, b)));
    }
  }

  @Test
  void combineIsAssociativeAcrossManyAppends() {
    // The commit path chains one combine per commit, so a long chain must still match one pass.
    SplittableRandom random = new SplittableRandom(99);
    byte[] all = new byte[0];
    long chained = crc(new byte[0]);
    for (int i = 0; i < 64; i++) {
      byte[] chunk = bytes(random, random.nextInt(1, 500));
      chained = Crc32Combine.combine(chained, crc(chunk), chunk.length);
      all = concat(all, chunk);
    }
    assertThat(chained).isEqualTo(crc(all));
  }

  @Test
  void anEmptyTailReturnsTheFirstCrc() {
    assertThat(Crc32Combine.combine(0x12345678L, 0xFFFFFFFFL, 0)).isEqualTo(0x12345678L);
  }

  @Test
  void aNegativeLengthIsRejected() {
    assertThatIllegalArgumentException().isThrownBy(() -> Crc32Combine.combine(1, 2, -1));
  }

  private static byte[] bytes(SplittableRandom random, int length) {
    byte[] out = new byte[length];
    for (int i = 0; i < length; i++) {
      out[i] = (byte) random.nextInt(256);
    }
    return out;
  }

  private static long crc(byte[] data) {
    CRC32 crc = new CRC32();
    crc.update(data);
    return crc.getValue();
  }

  private static byte[] concat(byte[] a, byte[] b) {
    byte[] out = new byte[a.length + b.length];
    System.arraycopy(a, 0, out, 0, a.length);
    System.arraycopy(b, 0, out, a.length, b.length);
    return out;
  }
}
