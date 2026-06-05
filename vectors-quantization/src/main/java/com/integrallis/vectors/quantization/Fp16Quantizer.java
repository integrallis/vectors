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
 * Half-precision (IEEE 754 binary16 / fp16) scalar quantizer. Stores each coordinate as a 16-bit
 * half-float — exactly half the footprint of {@code float32} — and scores by upcasting back to
 * {@code float32} on the fly (see {@link Fp16QuantizedVectors}).
 *
 * <p>Unlike {@link ScalarQuantizer}, there is nothing to train: the {@code float32 ↔ binary16}
 * conversion (via the stable {@link Float#floatToFloat16(float)} / {@link
 * Float#float16ToFloat(short)} APIs) is data-independent, so {@link #train(VectorDataset)} just
 * captures the dimension. fp16 keeps ~3–4 significant decimal digits and represents magnitudes up
 * to 65504, which is more than enough for the unit-norm embeddings typical of vector search, so
 * recall is effectively indistinguishable from full precision while halving storage and memory
 * bandwidth.
 *
 * <p>Coordinates are packed little-endian (two bytes per dimension) so the on-disk layout matches
 * the {@code LITTLE_ENDIAN} convention used by the rest of {@code vectors}.
 */
public final class Fp16Quantizer implements Quantizer<Fp16QuantizedVectors> {

  /** Bytes per coordinate in the half-precision encoding. */
  static final int BYTES_PER_DIM = Short.BYTES;

  private final int dimension;

  /**
   * Constructs a quantizer for the given dimension. Public so deserialization codecs can rebuild
   * the (state-free) quantizer directly from the stored dimension.
   *
   * @param dimension the vector dimension; must be positive
   */
  public Fp16Quantizer(int dimension) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
  }

  /**
   * "Trains" the quantizer — fp16 has no learned state, so this only reads the dataset dimension.
   *
   * @param dataset the source vectors (used only for its dimension)
   * @return a quantizer for {@code dataset.dimension()}
   */
  public static Fp16Quantizer train(VectorDataset dataset) {
    return new Fp16Quantizer(dataset.dimension());
  }

  @Override
  public byte[] encode(float[] vector) {
    byte[] dst = new byte[dimension * BYTES_PER_DIM];
    encode(vector, dst);
    return dst;
  }

  @Override
  public float encode(float[] vector, byte[] dst) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }
    for (int i = 0; i < dimension; i++) {
      short h = Float.floatToFloat16(vector[i]);
      int o = i << 1;
      dst[o] = (byte) (h & 0xFF);
      dst[o + 1] = (byte) ((h >>> 8) & 0xFF);
    }
    return 0f; // fp16 carries no per-vector correction term
  }

  @Override
  public float[] decode(byte[] encoded) {
    float[] out = new float[dimension];
    decodeInto(encoded, out, dimension);
    return out;
  }

  /**
   * Upcasts {@code dim} little-endian binary16 coordinates from {@code src} into {@code dst}.
   * Shared by {@link #decode(byte[])} and the {@link Fp16QuantizedVectors} scorer so both paths
   * agree bit-for-bit.
   *
   * @param src packed half-precision bytes ({@code >= dim * 2} bytes)
   * @param dst destination for the upcast {@code float32} values ({@code >= dim} long)
   * @param dim number of coordinates to decode
   */
  static void decodeInto(byte[] src, float[] dst, int dim) {
    for (int i = 0; i < dim; i++) {
      int o = i << 1;
      short h = (short) ((src[o] & 0xFF) | ((src[o + 1] & 0xFF) << 8));
      dst[i] = Float.float16ToFloat(h);
    }
  }

  @Override
  public Fp16QuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    byte[][] quantized = new byte[count][dimension * BYTES_PER_DIM];
    for (int i = 0; i < count; i++) {
      encode(dataset.getVector(i), quantized[i]);
    }
    return new Fp16QuantizedVectors(this, quantized, dimension);
  }

  @Override
  public float compressionRatio() {
    return (float) Float.BYTES / BYTES_PER_DIM; // 4 bytes -> 2 bytes = 2.0x
  }

  @Override
  public int dimension() {
    return dimension;
  }
}
