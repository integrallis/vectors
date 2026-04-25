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
package com.integrallis.vectors.quantization;

/**
 * Bit-width configurations for scalar quantization.
 *
 * <ul>
 *   <li>{@link #INT8} — 7-bit unsigned [0, 127], one value per byte, 4x compression
 *   <li>{@link #INT4} — 4-bit unsigned [0, 15], two values per byte (packed nibbles), 8x
 *       compression
 * </ul>
 */
public enum ScalarBits {

  /** 7-bit unsigned quantization. Values in [0, 127], one per byte. 4x compression vs float32. */
  INT8(7, 127, 4.0f),

  /** 4-bit unsigned quantization. Values in [0, 15], packed two per byte. 8x compression. */
  INT4(4, 15, 8.0f);

  private final int bits;
  private final int maxValue;
  private final float compressionRatio;

  ScalarBits(int bits, int maxValue, float compressionRatio) {
    this.bits = bits;
    this.maxValue = maxValue;
    this.compressionRatio = compressionRatio;
  }

  /** Returns the number of bits per dimension. */
  public int bits() {
    return bits;
  }

  /** Returns the maximum quantized value (e.g., 127 for INT8, 15 for INT4). */
  public int maxValue() {
    return maxValue;
  }

  /** Returns the compression ratio vs float32 (4.0 for INT8, 8.0 for INT4). */
  public float compressionRatio() {
    return compressionRatio;
  }

  /**
   * Returns the encoded byte array size for a vector with the given dimension.
   *
   * @param dimension the vector dimensionality
   * @return the number of bytes in the encoded representation
   */
  public int encodedByteSize(int dimension) {
    return switch (this) {
      case INT8 -> dimension;
      case INT4 -> (dimension + 1) / 2; // ceil(dim/2): two nibbles per byte
    };
  }
}
