package com.integrallis.vectors.storage;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Value layout constants for on-disk and off-heap storage. All layouts use little-endian byte order
 * (native for x86-64 and ARM in standard configuration) and unaligned access (handles vectors whose
 * dimension is not a multiple of the SIMD width).
 */
public final class StorageLayouts {

  private StorageLayouts() {}

  /** Little-endian 32-bit float, unaligned access. */
  public static final ValueLayout.OfFloat FLOAT_LE =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  /** Little-endian 32-bit integer, unaligned access. */
  public static final ValueLayout.OfInt INT_LE =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  /** Little-endian 64-bit long, unaligned access. */
  public static final ValueLayout.OfLong LONG_LE =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  /** Little-endian 16-bit short, unaligned access. */
  public static final ValueLayout.OfShort SHORT_LE =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  /** Single byte layout. */
  public static final ValueLayout.OfByte BYTE = ValueLayout.JAVA_BYTE;
}
