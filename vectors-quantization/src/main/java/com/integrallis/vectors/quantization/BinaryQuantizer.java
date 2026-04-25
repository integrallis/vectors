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
 * Binary quantization of float32 vectors to 1-bit-per-dimension encoding.
 *
 * <p>Supports two modes via {@link BinaryMode}:
 *
 * <ul>
 *   <li><b>SIGN_BIT</b>: Simple sign-bit encoding ({@code bit[d] = v[d] > 0 ? 1 : 0}). No training
 *       required. Scoring uses Hamming distance. 32x compression.
 *   <li><b>BBQ</b> (Better Binary Quantization): Centroid-relative encoding ({@code bit[d] = (v[d]
 *       - centroid[d]) >= 0 ? 1 : 0}) with per-vector correction factors for asymmetric int4-query
 *       x 1-bit-stored scoring. 32x compression + 12 bytes overhead per vector. Derived from RaBitQ
 *       / Lucene BBQ.
 * </ul>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * BinaryQuantizer bq = BinaryQuantizer.train(data);                   // SIGN_BIT
 * BinaryQuantizer bbq = BinaryQuantizer.train(data, BinaryMode.BBQ);  // BBQ
 * BinaryQuantizedVectors compressed = bq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
 * }</pre>
 */
public final class BinaryQuantizer implements Quantizer<BinaryQuantizedVectors> {

  private final int dimension;
  private final BinaryMode mode;
  private final float[] centroid; // null for SIGN_BIT, dataset mean for BBQ

  private BinaryQuantizer(int dimension, BinaryMode mode, float[] centroid) {
    this.dimension = dimension;
    this.mode = mode;
    this.centroid = centroid;
  }

  // --- Factory methods ---

  /**
   * Reconstructs a trained {@code BinaryQuantizer} from previously serialized state. Used by
   * deserialization codecs to restore a quantizer without re-computing the centroid.
   *
   * @param dimension the original vector dimension
   * @param mode SIGN_BIT or BBQ
   * @param centroid the dataset centroid (null for SIGN_BIT, non-null for BBQ)
   * @return a reconstructed binary quantizer
   */
  public static BinaryQuantizer fromState(int dimension, BinaryMode mode, float[] centroid) {
    return new BinaryQuantizer(dimension, mode, centroid);
  }

  /**
   * Trains a SIGN_BIT binary quantizer (no centroid, no training needed).
   *
   * @param dataset the source data (used only for dimension)
   * @return a SIGN_BIT binary quantizer
   */
  public static BinaryQuantizer train(VectorDataset dataset) {
    return train(dataset, BinaryMode.SIGN_BIT);
  }

  /**
   * Trains a binary quantizer with the specified mode.
   *
   * @param dataset the training data
   * @param mode SIGN_BIT or BBQ
   * @return a trained binary quantizer
   */
  public static BinaryQuantizer train(VectorDataset dataset, BinaryMode mode) {
    int dim = dataset.dimension();
    return switch (mode) {
      case SIGN_BIT -> new BinaryQuantizer(dim, BinaryMode.SIGN_BIT, null);
      case BBQ -> new BinaryQuantizer(dim, BinaryMode.BBQ, dataset.computeCentroid());
    };
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
    return packBits(vector, dst, centroid);
  }

  @Override
  public float[] decode(byte[] encoded) {
    float[] result = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      int byteIndex = d >> 3;
      int bitIndex = d & 7;
      boolean bit = ((encoded[byteIndex] >> bitIndex) & 1) == 1;
      result[d] = bit ? 1.0f : -1.0f;
    }
    if (centroid != null) {
      VectorUtil.addInPlace(result, centroid);
    }
    return result;
  }

  @Override
  public BinaryQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    int numLongs = numLongs();
    long[][] codes = new long[count][numLongs];
    float[][] corrections = mode == BinaryMode.BBQ ? new float[count][3] : null;

    byte[] buf = new byte[encodedByteSize()];
    for (int i = 0; i < count; i++) {
      float[] vector = dataset.getVector(i);
      // packBits returns distToC as a by-product of its centered-value loop; reuse it so
      // computeCorrections only needs a single additional pass for min/max (not distToC).
      float distToC = packBits(vector, buf, centroid);
      bytesToLongs(buf, codes[i]);

      if (corrections != null) {
        computeCorrections(vector, centroid, distToC, corrections[i]);
      }
    }

    return new BinaryQuantizedVectors(this, codes, corrections, dataset.dimension());
  }

  @Override
  public float compressionRatio() {
    return (dimension * 4.0f) / totalEncodedByteSize();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors ---

  /** Returns the binary quantization mode. */
  public BinaryMode mode() {
    return mode;
  }

  /** Returns the centroid (dataset mean), or null if mode is SIGN_BIT. */
  public float[] centroid() {
    return centroid;
  }

  /**
   * Returns the number of bytes for the bit-code portion of one encoded vector (ceil(dimension /
   * 8)). For BBQ mode, 12 additional bytes of correction floats are stored separately; use {@link
   * #totalEncodedByteSize()} to get the full per-vector footprint.
   */
  int encodedByteSize() {
    return (dimension + 7) >> 3;
  }

  /**
   * Returns the total per-vector storage in bytes, including correction factors for BBQ mode.
   * Equivalent to {@link #encodedByteSize()} for SIGN_BIT (no corrections).
   */
  int totalEncodedByteSize() {
    return encodedByteSize() + (mode == BinaryMode.BBQ ? 12 : 0);
  }

  /** Returns the number of longs needed to store one encoded vector (ceil(dimension / 64)). */
  int numLongs() {
    return (dimension + 63) >> 6;
  }

  // --- Internal helpers ---

  /**
   * Sign-bit encodes a vector into packed bytes. If centroid is non-null, the vector is centered
   * first. Returns the distToC correction (squared norm of centered vector) for BBQ, or 0 for
   * SIGN_BIT.
   *
   * <p>The destination buffer is zero-initialized before packing to prevent stale bits from a
   * reused buffer.
   */
  static float packBits(float[] vector, byte[] dst, float[] centroid) {
    Arrays.fill(dst, (byte) 0); // zero-init: OR-packing requires clean slate
    float distToC = 0f;
    for (int d = 0; d < vector.length; d++) {
      float v = centroid != null ? vector[d] - centroid[d] : vector[d];
      if (centroid != null) {
        distToC += v * v;
      }
      if (v >= 0) {
        int byteIndex = d >> 3;
        int bitIndex = d & 7;
        dst[byteIndex] |= (byte) (1 << bitIndex);
      }
    }
    return distToC;
  }

  /** Converts packed bytes to longs for efficient Hamming distance via {@code VectorUtil}. */
  static void bytesToLongs(byte[] bytes, long[] longs) {
    Arrays.fill(longs, 0L);
    for (int i = 0; i < bytes.length; i++) {
      int longIndex = i >> 3;
      int shift = (i & 7) << 3;
      longs[longIndex] |= (bytes[i] & 0xFFL) << shift;
    }
  }

  /** Converts longs back to packed bytes. */
  static void longsToBytes(long[] longs, byte[] bytes) {
    Arrays.fill(bytes, (byte) 0);
    for (int i = 0; i < bytes.length; i++) {
      int longIndex = i >> 3;
      int shift = (i & 7) << 3;
      bytes[i] = (byte) ((longs[longIndex] >> shift) & 0xFF);
    }
  }

  /**
   * Computes per-vector correction factors for BBQ scoring.
   *
   * <p>{@code distToC} (||v - centroid||²) is passed in from the caller who already computed it as
   * a by-product of {@link #packBits}, avoiding a redundant full-vector pass.
   *
   * @param vector the original float vector
   * @param centroid the dataset centroid
   * @param distToC pre-computed squared norm of (vector - centroid)
   * @param out output array of length 3: [distToC, vl, width]
   */
  static void computeCorrections(float[] vector, float[] centroid, float distToC, float[] out) {
    float min = Float.POSITIVE_INFINITY;
    float max = Float.NEGATIVE_INFINITY;
    for (int d = 0; d < vector.length; d++) {
      float v = vector[d] - centroid[d];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    out[0] = distToC;
    out[1] = min; // vl: lower bound
    out[2] = max - min; // width: range
  }
}
