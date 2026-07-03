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
package com.integrallis.vectors.storage.memory;

/**
 * Alignment helpers for SIMD-friendly memory layout. Provides constants for page, cache-line, and
 * vector-register alignment, plus rounding utilities.
 */
public final class AlignmentUtil {

  private AlignmentUtil() {}

  /**
   * Portable logical page size used for section alignment in on-disk formats. Override with {@code
   * -Dvectors.pageSize=<power-of-two>} when a different format alignment is required.
   */
  public static final int PAGE_SIZE = configuredPageSize();

  /**
   * Cache-line and SIMD vector alignment (64 bytes). Compatible with x86 cache lines (64B), ARM
   * Neoverse cache lines (64B), and AVX-512 registers (64B).
   */
  public static final int VECTOR_ALIGNMENT = 64;

  /**
   * Rounds {@code value} up to the next multiple of {@code alignment}. Alignment must be a power of
   * two.
   *
   * @param value the value to align
   * @param alignment the alignment boundary (must be a power of two)
   * @return the aligned value
   * @throws IllegalArgumentException if alignment is not a positive power of two
   */
  public static long alignUp(long value, int alignment) {
    if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
      throw new IllegalArgumentException("Alignment must be a positive power of two: " + alignment);
    }
    long mask = alignment - 1L;
    return (value + mask) & ~mask;
  }

  /**
   * Returns the number of padding bytes needed to align {@code offset} to the given boundary.
   *
   * @param offset the current offset
   * @param alignment the alignment boundary (must be a power of two)
   * @return the number of padding bytes (0 if already aligned)
   */
  public static int paddingFor(long offset, int alignment) {
    return (int) (alignUp(offset, alignment) - offset);
  }

  /**
   * Returns true if {@code value} is aligned to the given boundary.
   *
   * @param value the value to check
   * @param alignment the alignment boundary (must be a power of two)
   * @return true if aligned
   */
  public static boolean isAligned(long value, int alignment) {
    return (value & (alignment - 1L)) == 0;
  }

  private static int configuredPageSize() {
    String override = System.getProperty("vectors.pageSize");
    return requirePowerOfTwoPageSize(
        override == null ? 4096 : Integer.parseInt(override), "vectors.pageSize");
  }

  private static int requirePowerOfTwoPageSize(int pageSize, String source) {
    if (pageSize <= 0 || (pageSize & (pageSize - 1)) != 0) {
      throw new IllegalStateException(source + " returned invalid page size: " + pageSize);
    }
    return pageSize;
  }
}
