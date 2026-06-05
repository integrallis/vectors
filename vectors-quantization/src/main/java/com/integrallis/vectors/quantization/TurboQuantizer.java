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

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;

/**
 * TurboQuant quantizer — rotation-based MSE-optimal per-coordinate quantization at any bit-width.
 * Implements the MSE stage of the TurboQuant algorithm (ICLR 2026, Google) with pluggable rotation
 * strategies from RotorQuant.
 *
 * <p><b>Encoding pipeline:</b> center → normalize → rotate → per-coordinate Lloyd-Max quantize →
 * pack indices. The per-vector norm is stored as a correction factor for distance estimation.
 *
 * <p><b>Rotation strategies</b> (via {@link Rotation}):
 *
 * <ul>
 *   <li>{@link RandomRotation}: Dense d×d QR matrix (O(d²) FMAs) — TurboQuant original
 *   <li>{@link GivensRotation}: 2D pair Givens (O(d) FMAs) — PlanarQuant, fastest
 *   <li>{@link QuaternionRotation}: 4D block quaternion (O(d) FMAs) — IsoQuant, best quality
 * </ul>
 *
 * <p><b>Key theoretical property:</b> After random rotation, each coordinate of a unit vector
 * follows approximately N(0, 1/d). The {@link LloydMaxCodebook} provides MSE-optimal scalar
 * quantization for this distribution, achieving near-optimal distortion within 2.7× of the Shannon
 * lower bound.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * TurboQuantizer tq = TurboQuantizer.train(data, 4); // 4-bit, default Givens rotation
 * TurboQuantizedVectors compressed = tq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
 * }</pre>
 *
 * @see Rotation
 * @see LloydMaxCodebook
 * @see TurboQuantizedVectors
 */
public final class TurboQuantizer implements Quantizer<TurboQuantizedVectors> {

  private final int dimension;
  private final int paddedDimension;
  private final int bits;
  private final float[] centroid;
  private final float[] paddedCentroid;
  private final Rotation rotation;
  private final LloydMaxCodebook codebook;

  private TurboQuantizer(
      int dimension,
      int paddedDimension,
      int bits,
      float[] centroid,
      Rotation rotation,
      LloydMaxCodebook codebook) {
    this.dimension = dimension;
    this.paddedDimension = paddedDimension;
    this.bits = bits;
    this.centroid = centroid;
    this.paddedCentroid = new float[paddedDimension];
    System.arraycopy(centroid, 0, this.paddedCentroid, 0, dimension);
    this.rotation = rotation;
    this.codebook = codebook;
  }

  // --- Factory methods ---

  /**
   * Trains a TurboQuantizer with the default rotation strategy ({@link GivensRotation}).
   *
   * @param dataset the training data
   * @param bits quantization bit-width (1-8)
   * @return a trained TurboQuantizer
   */
  public static TurboQuantizer train(VectorDataset dataset, int bits) {
    return train(dataset, bits, 42L);
  }

  /**
   * Trains a TurboQuantizer with the default rotation strategy and specified seed.
   *
   * @param dataset the training data
   * @param bits quantization bit-width (1-8)
   * @param seed random seed for rotation generation
   * @return a trained TurboQuantizer
   */
  public static TurboQuantizer train(VectorDataset dataset, int bits, long seed) {
    int dim = dataset.dimension();
    int paddedDim = ((dim + 63) / 64) * 64;
    Rotation rotation = GivensRotation.generate(paddedDim, seed);
    return train(dataset, bits, rotation);
  }

  /**
   * Trains a TurboQuantizer with a pluggable rotation strategy.
   *
   * @param dataset the training data
   * @param bits quantization bit-width (1-8)
   * @param rotation the rotation strategy
   * @return a trained TurboQuantizer
   * @throws IllegalArgumentException if rotation dimension doesn't match padded dimension
   */
  public static TurboQuantizer train(VectorDataset dataset, int bits, Rotation rotation) {
    int dim = dataset.dimension();
    int paddedDim = ((dim + 63) / 64) * 64;
    if (rotation.dimension() != paddedDim) {
      throw new IllegalArgumentException(
          "Rotation dimension "
              + rotation.dimension()
              + " must match padded dimension "
              + paddedDim);
    }
    float[] centroid = dataset.computeCentroid();
    LloydMaxCodebook codebook = LloydMaxCodebook.compute(paddedDim, bits);
    return new TurboQuantizer(dim, paddedDim, bits, centroid, rotation, codebook);
  }

  /**
   * Reconstructs a quantizer from its serialized state (used by the persistence codec). The
   * Lloyd-Max codebook is not stored — it is data-independent (MSE-optimal for the {@code N(0,
   * 1/paddedDimension)} per-coordinate distribution) and recomputed here from {@code
   * (paddedDimension, bits)}, exactly as {@link #train} computed it.
   *
   * @param dimension original (unpadded) vector dimension
   * @param paddedDimension padded dimension; must match {@code rotation.dimension()}
   * @param bits quantization bit-width (1-8)
   * @param centroid the training centroid (length {@code dimension})
   * @param rotation the rotation strategy (its dimension must equal {@code paddedDimension})
   * @return a quantizer equivalent to the one whose state was serialized
   * @throws IllegalArgumentException if the rotation dimension does not match {@code
   *     paddedDimension}
   */
  public static TurboQuantizer fromState(
      int dimension, int paddedDimension, int bits, float[] centroid, Rotation rotation) {
    if (rotation.dimension() != paddedDimension) {
      throw new IllegalArgumentException(
          "Rotation dimension "
              + rotation.dimension()
              + " must match padded dimension "
              + paddedDimension);
    }
    LloydMaxCodebook codebook = LloydMaxCodebook.compute(paddedDimension, bits);
    return new TurboQuantizer(dimension, paddedDimension, bits, centroid, rotation, codebook);
  }

  // --- Quantizer interface ---

  @Override
  public byte[] encode(float[] vector) {
    byte[] dst = new byte[encodedByteSize()];
    encode(vector, dst);
    return dst;
  }

  @Override
  public float encode(float[] vector, byte[] dst) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }

    // Center and pad
    float[] padded = centerAndPad(vector);

    // Compute norm before normalization
    float sqrX = VectorUtil.dotProduct(padded, padded);
    float norm = (float) Math.sqrt(sqrX);

    // Normalize to unit vector
    if (norm > 0) {
      VectorUtil.scale(padded, 1.0f / norm);
    }

    // Rotate
    float[] rotated = rotation.rotate(padded);

    // Per-coordinate Lloyd-Max quantize and pack into dst
    packIndices(rotated, dst);

    return norm; // correction factor
  }

  /**
   * Decodes a packed byte array back to an approximate vector.
   *
   * <p><b>Important limitation:</b> The per-vector norm (stored separately in {@link
   * TurboQuantizedVectors}) is not embedded in the byte array, so this method can only reconstruct
   * the <em>unit-direction</em> approximation of {@code (v - centroid)}, shifted back by the
   * centroid. The magnitude of {@code (v - centroid)} is lost.
   *
   * <p>Callers that require a full reconstruction (direction + scale) must use {@link
   * #reconstructCentered(byte[], float)} and add the centroid manually.
   *
   * @param encoded the packed index bytes produced by {@link #encode(float[])}
   * @return an approximate reconstruction (direction only — magnitude is not preserved)
   */
  @Override
  public float[] decode(byte[] encoded) {
    // Unpack indices and dequantize
    float[] rotated = new float[paddedDimension];
    unpackAndDequantize(encoded, rotated);

    // Inverse rotate
    float[] unrotated = rotation.inverseRotate(rotated);

    // Truncate to original dimension and add centroid.
    // Without the stored norm, we reconstruct only the unit-vector direction of (v − centroid).
    float[] result = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      result[d] = unrotated[d] + centroid[d];
    }
    return result;
  }

  @Override
  public TurboQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    byte[][] allIndices = new byte[count][];
    float[] allNorms = new float[count];

    for (int i = 0; i < count; i++) {
      allIndices[i] = new byte[encodedByteSize()];
      allNorms[i] = encode(dataset.getVector(i), allIndices[i]);
    }

    return new TurboQuantizedVectors(this, allIndices, allNorms, dimension);
  }

  @Override
  public float compressionRatio() {
    // Original: dimension * 4 bytes (float32)
    // Compressed: encodedByteSize() + 4 bytes (norm float)
    int compressedBytes = encodedByteSize() + Float.BYTES;
    return (dimension * (float) Float.BYTES) / compressedBytes;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors (public ones expose the serializable state for the db codec) ---

  /** Returns the centroid (length {@code dimension}). */
  public float[] centroid() {
    return centroid;
  }

  /** Returns the rotation strategy. */
  public Rotation rotation() {
    return rotation;
  }

  /** Returns the Lloyd-Max codebook. */
  LloydMaxCodebook codebook() {
    return codebook;
  }

  /** Returns the padded dimension (next multiple of 64 at or above {@code dimension}). */
  public int paddedDimension() {
    return paddedDimension;
  }

  /** Returns the quantization bit-width (1-8). */
  public int bits() {
    return bits;
  }

  /**
   * Reconstructs the centered (not adding centroid) approximate vector from stored indices and
   * norm. Used by {@link TurboQuantizedVectors} for scoring.
   */
  float[] reconstructCentered(byte[] packedIndices, float norm) {
    // Dequantize in rotated space
    float[] rotated = new float[paddedDimension];
    unpackAndDequantize(packedIndices, rotated);

    // Inverse rotate
    float[] unrotated = rotation.inverseRotate(rotated);

    // Scale by stored norm to get centered vector (not unit vector)
    float[] result = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      result[d] = unrotated[d] * norm;
    }
    return result;
  }

  // --- Internal helpers ---

  /** Centers and pads a vector: subtracts centroid and zero-pads to paddedDimension. */
  private float[] centerAndPad(float[] vector) {
    float[] padded = new float[paddedDimension];
    System.arraycopy(vector, 0, padded, 0, dimension);
    VectorUtil.subInPlace(padded, paddedCentroid);
    return padded;
  }

  /** Returns the number of bytes needed to store one encoded vector's indices. */
  public int encodedByteSize() {
    // Each dimension gets `bits` bits; total bits = paddedDimension * bits
    return (paddedDimension * bits + 7) / 8;
  }

  /**
   * Quantizes each coordinate using the Lloyd-Max codebook and packs indices into bytes. Indices
   * are packed LSB-first: for bits=4, byte[0] holds indices[0] in the lower nibble (bits 0-3) and
   * indices[1] in the upper nibble (bits 4-7).
   */
  private void packIndices(float[] rotated, byte[] dst) {
    Arrays.fill(dst, (byte) 0);
    int bitPos = 0;
    for (int d = 0; d < paddedDimension; d++) {
      int idx = codebook.quantize(rotated[d]);
      // Pack idx (bits wide) starting at bitPos
      int byteIdx = bitPos / 8;
      int bitOffset = bitPos % 8;

      if (bitOffset + bits <= 8) {
        // Fits in one byte
        dst[byteIdx] |= (byte) (idx << bitOffset);
      } else {
        // Spans two bytes
        dst[byteIdx] |= (byte) (idx << bitOffset);
        dst[byteIdx + 1] |= (byte) (idx >> (8 - bitOffset));
      }
      bitPos += bits;
    }
  }

  /** Unpacks indices from bytes and dequantizes each coordinate using the Lloyd-Max codebook. */
  private void unpackAndDequantize(byte[] packed, float[] rotated) {
    int mask = (1 << bits) - 1;
    int bitPos = 0;
    for (int d = 0; d < paddedDimension; d++) {
      int byteIdx = bitPos / 8;
      int bitOffset = bitPos % 8;

      int idx;
      if (bitOffset + bits <= 8) {
        idx = ((packed[byteIdx] & 0xFF) >> bitOffset) & mask;
      } else {
        // Spans two bytes
        int lo = (packed[byteIdx] & 0xFF) >>> bitOffset;
        int hi = (packed[byteIdx + 1] & 0xFF) << (8 - bitOffset);
        idx = (lo | hi) & mask;
      }
      rotated[d] = codebook.dequantize(idx);
      bitPos += bits;
    }
  }
}
