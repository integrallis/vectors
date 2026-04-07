package com.integrallis.vectors.storage.memory;

/**
 * Alignment helpers for SIMD-friendly memory layout. Provides constants for page, cache-line, and
 * vector-register alignment, plus rounding utilities.
 */
public final class AlignmentUtil {

  private AlignmentUtil() {}

  /** Default OS page size (4KB). Used for section-level alignment in on-disk formats. */
  public static final int PAGE_SIZE = detectPageSize();

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

  private static int detectPageSize() {
    // Try to get the actual page size from the system. This is typically 4096 on x86/ARM.
    // JDK doesn't expose sysconf(_SC_PAGESIZE) directly, but on most systems it's 4096.
    // We could use FFM Linker to call sysconf, but for now use the safe default.
    String override = System.getProperty("vectors.pageSize");
    if (override != null) {
      return Integer.parseInt(override);
    }
    return 4096;
  }
}
