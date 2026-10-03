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

/**
 * Combines two CRC-32 values into the CRC of their concatenation, without reading either input
 * again.
 *
 * <p>This is what lets a commit keep a CRC over a file it only appended to. {@code vectors.bin}
 * grows by the staged vectors and nothing else, so the successor's checksum is {@code
 * combine(predecessorCrc, crcOfAppendedBytes, appendedLength)} — O(appended) work instead of
 * re-reading the whole file, which is the difference between a commit costing the batch and a
 * commit costing the collection.
 *
 * <p>{@link java.util.zip.CRC32} exposes no equivalent, so this is the standard GF(2) construction
 * used by zlib's {@code crc32_combine64}: CRC-32 is linear over GF(2), so appending {@code n} zero
 * bytes to a message is a fixed linear operator on its CRC, and that operator is applied by
 * square-and-multiply over the bits of {@code n}.
 *
 * <p>Correctness is not argued from the derivation — {@code Crc32CombineTest} checks it against
 * {@link java.util.zip.CRC32} over concatenated inputs, including empty and single-byte tails and
 * lengths spanning many powers of two.
 */
public final class Crc32Combine {

  /** The CRC-32 polynomial in reflected form, as used by {@link java.util.zip.CRC32}. */
  private static final int POLYNOMIAL = 0xEDB88320;

  private Crc32Combine() {}

  /**
   * CRC-32 of {@code a || b}, given each part's CRC and the length of the second.
   *
   * @param crcA CRC-32 of the first part
   * @param crcB CRC-32 of the second part
   * @param lengthB length in bytes of the second part; zero returns {@code crcA}
   * @throws IllegalArgumentException if {@code lengthB} is negative
   */
  public static long combine(long crcA, long crcB, long lengthB) {
    if (lengthB < 0) {
      throw new IllegalArgumentException("lengthB must not be negative: " + lengthB);
    }
    if (lengthB == 0) {
      return crcA & 0xFFFFFFFFL;
    }

    // even[] holds the operator for 2^k zero bytes, odd[] for 2^(k+1); they alternate as the
    // square-and-multiply walks the bits of lengthB.
    int[] even = new int[32];
    int[] odd = new int[32];

    // odd starts as the operator for one zero bit: the polynomial, then the identity basis.
    odd[0] = POLYNOMIAL;
    int row = 1;
    for (int i = 1; i < 32; i++) {
      odd[i] = row;
      row <<= 1;
    }

    square(even, odd); // even = one zero byte's worth? no: two zero bits
    square(odd, even); // odd  = four zero bits

    int crc = (int) crcA;
    long remaining = lengthB;
    do {
      square(even, odd);
      if ((remaining & 1) != 0) {
        crc = apply(even, crc);
      }
      remaining >>>= 1;
      if (remaining == 0) {
        break;
      }
      square(odd, even);
      if ((remaining & 1) != 0) {
        crc = apply(odd, crc);
      }
      remaining >>>= 1;
    } while (remaining != 0);

    return (crc ^ (int) crcB) & 0xFFFFFFFFL;
  }

  /** Applies a GF(2) matrix (as 32 column vectors) to a vector. */
  private static int apply(int[] matrix, int vector) {
    int sum = 0;
    int index = 0;
    while (vector != 0) {
      if ((vector & 1) != 0) {
        sum ^= matrix[index];
      }
      vector >>>= 1;
      index++;
    }
    return sum;
  }

  /** {@code into = square(from)}: the operator for twice as many zero bytes. */
  private static void square(int[] into, int[] from) {
    for (int i = 0; i < 32; i++) {
      into[i] = apply(from, from[i]);
    }
  }
}
