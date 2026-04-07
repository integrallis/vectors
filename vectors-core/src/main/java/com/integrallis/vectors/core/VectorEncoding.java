package com.integrallis.vectors.core;

/**
 * Supported vector element encodings. Each encoding defines the byte width per dimension and the
 * Java type used for distance computations.
 *
 * <ul>
 *   <li>{@link #FLOAT32} — 4 bytes per dimension, full-precision IEEE 754 float
 *   <li>{@link #INT8} — 1 byte per dimension, signed byte [-128, 127]
 *   <li>{@link #BINARY} — 1 bit per dimension, packed into long arrays
 * </ul>
 */
public enum VectorEncoding {

  /** 32-bit IEEE 754 floating point. 4 bytes per dimension. */
  FLOAT32(Float.BYTES),

  /** Signed 8-bit integer. 1 byte per dimension. */
  INT8(Byte.BYTES),

  /**
   * Binary encoding. 1 bit per dimension, packed into {@code long[]} arrays. The byte size per
   * dimension is not meaningful for binary; use {@link #vectorByteSize(int)} instead.
   */
  BINARY(0);

  private final int bytesPerDimension;

  VectorEncoding(int bytesPerDimension) {
    this.bytesPerDimension = bytesPerDimension;
  }

  /** Returns the number of bytes per dimension element, or 0 for {@link #BINARY}. */
  public int bytesPerDimension() {
    return bytesPerDimension;
  }

  /**
   * Returns the total byte size required to store a vector with the given number of dimensions.
   *
   * @param dimensions the number of dimensions
   * @return byte size of the vector data
   */
  public int vectorByteSize(int dimensions) {
    return switch (this) {
      case FLOAT32 -> dimensions * Float.BYTES;
      case INT8 -> dimensions;
      case BINARY -> {
        // Ceil division: (dimensions + 63) / 64 longs, each 8 bytes
        int longs = (dimensions + Long.SIZE - 1) / Long.SIZE;
        yield longs * Long.BYTES;
      }
    };
  }
}
