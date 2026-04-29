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
 * Extended RaBitQ quantizer (SIGMOD 2025). Extends 1-bit RaBitQ to arbitrary bit-widths (2-8
 * bits/dim) by storing both sign bits and magnitude codes.
 *
 * <p><b>Key advantage over TurboQuant:</b> Correction-factor-based scoring with zero per-score
 * allocation, making it suitable for high-throughput ANN search inside graph traversal. TurboQuant
 * requires reconstruction (3 float[] allocations per score call).
 *
 * <p><b>Encoding pipeline:</b> center → pad → normalize → rotate → sign-bit encode → greedy
 * magnitude quantize → pack. Per-vector correction factors enable unbiased distance estimation.
 *
 * <p><b>Two-layer scoring:</b>
 *
 * <ol>
 *   <li>Layer 1 (1-bit coarse): identical to 1-bit RaBitQ — cheap popcount-based estimate
 *   <li>Layer 2 (multi-bit refinement): uses magnitude codes for refined distance estimate
 * </ol>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * ExtendedRaBitQuantizer eq = ExtendedRaBitQuantizer.train(data, 4); // 4-bit
 * ExtendedRaBitQuantizedVectors compressed = eq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
 * }</pre>
 *
 * @see RaBitQuantizer
 * @see ExtendedRaBitQuantizedVectors
 * @see Rotation
 */
public final class ExtendedRaBitQuantizer implements Quantizer<ExtendedRaBitQuantizedVectors> {

  /** Number of per-vector correction floats: sqrX, x0, factorPpc, factorIp, errorFactor. */
  static final int NUM_CORRECTIONS = 6;

  /** Correction index: squared distance to centroid ||v - centroid||². */
  static final int IDX_SQR_X = 0;

  /** Correction index: inner product between unit residual and its quantized form. */
  static final int IDX_X0 = 1;

  /** Correction index: pre-computed partial popcount correction factor. */
  static final int IDX_FACTOR_PPC = 2;

  /** Correction index: scaling factor converting binary IP to approximate distance. */
  static final int IDX_FACTOR_IP = 3;

  /** Correction index: error bound factor for optional filtering. */
  static final int IDX_ERROR_FACTOR = 4;

  /**
   * Correction index: {@code 1 / (x0Multi * normCode)} — decouples the multi-bit norm product from
   * {@code factorIpMulti} so the scorer can inject per-query magnitude without re-deriving code
   * norms from packed magnitudes.
   */
  static final int IDX_XIP_NORM = 5;

  /** Confidence multiplier for error bound (≈ 1.9σ covers ~97% of estimation errors). */
  private static final float ERROR_CONFIDENCE = 1.9f;

  /** Number of candidate rescale factors to evaluate during greedy magnitude quantization. */
  private static final int NUM_RESCALE_CANDIDATES = 50;

  private final int dimension;
  private final int paddedDimension;
  private final int bits;
  private final float[] centroid;
  private final float[] paddedCentroid;
  private final Rotation rotation;

  private ExtendedRaBitQuantizer(
      int dimension, int paddedDimension, int bits, float[] centroid, Rotation rotation) {
    this.dimension = dimension;
    this.paddedDimension = paddedDimension;
    this.bits = bits;
    this.centroid = centroid;
    this.paddedCentroid = new float[paddedDimension];
    System.arraycopy(centroid, 0, this.paddedCentroid, 0, dimension);
    this.rotation = rotation;
  }

  // --- Factory methods ---

  /**
   * Trains an Extended RaBitQ quantizer with default seed (42L) and default rotation ({@link
   * GivensRotation}).
   *
   * @param dataset the training data
   * @param bits magnitude bit-width (2-8; use {@link RaBitQuantizer} for 1-bit)
   * @return a trained Extended RaBitQ quantizer
   * @throws IllegalArgumentException if bits is not in [2, 8]
   */
  public static ExtendedRaBitQuantizer train(VectorDataset dataset, int bits) {
    return train(dataset, bits, 42L);
  }

  /**
   * Trains an Extended RaBitQ quantizer with a Givens rotation (O(d) FMAs).
   *
   * <p>For an explicit dense rotation, use {@link #train(VectorDataset, int, Rotation)} with {@link
   * RandomRotation#generate}.
   *
   * @param dataset the training data
   * @param bits magnitude bit-width (2-8)
   * @param seed random seed for the rotation
   * @return a trained Extended RaBitQ quantizer
   * @throws IllegalArgumentException if bits is not in [2, 8]
   */
  public static ExtendedRaBitQuantizer train(VectorDataset dataset, int bits, long seed) {
    validateBits(bits);
    int dim = dataset.dimension();
    int paddedDim = ((dim + 63) / 64) * 64;
    Rotation rotation = GivensRotation.generate(paddedDim, seed);
    return train(dataset, bits, rotation);
  }

  /**
   * Trains an Extended RaBitQ quantizer with a pluggable rotation strategy.
   *
   * @param dataset the training data
   * @param bits magnitude bit-width (2-8)
   * @param rotation the rotation strategy (must have dimension == padded dimension)
   * @return a trained Extended RaBitQ quantizer
   * @throws IllegalArgumentException if bits is not in [2, 8] or rotation dimension mismatches
   */
  public static ExtendedRaBitQuantizer train(VectorDataset dataset, int bits, Rotation rotation) {
    validateBits(bits);
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
    return new ExtendedRaBitQuantizer(dim, paddedDim, bits, centroid, rotation);
  }

  private static void validateBits(int bits) {
    if (bits < 2 || bits > 8) {
      throw new IllegalArgumentException(
          "Extended RaBitQ bits must be in [2, 8], got "
              + bits
              + ". Use RaBitQuantizer for 1-bit.");
    }
  }

  // --- Quantizer interface ---

  @Override
  public byte[] encode(float[] vector) {
    byte[] dst = new byte[encodedByteSize()];
    encode(vector, dst);
    return dst;
  }

  /**
   * Encodes a single vector into the destination byte array.
   *
   * <p>Layout: [sign bits (signByteSize bytes)] [magnitude codes (magByteSize bytes)].
   *
   * <p>Pipeline: center → pad → compute norm² → normalize → rotate → sign-bit pack → greedy
   * magnitude quantize → pack magnitudes.
   *
   * @param vector the input float vector (must have length == dimension)
   * @param dst the destination byte array (must have length >= encodedByteSize())
   * @return the squared distance to centroid (||v - centroid||²)
   */
  @Override
  public float encode(float[] vector, byte[] dst) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }
    int required = encodedByteSize();
    if (dst.length < required) {
      throw new IllegalArgumentException("dst length " + dst.length + " < required " + required);
    }

    // Center and pad
    float[] padded = centerAndPad(vector);

    // Compute ||v - centroid||² before normalization
    float sqrX = VectorUtil.dotProduct(padded, padded);

    // Normalize to unit vector
    if (sqrX > 0) {
      VectorUtil.scale(padded, 1.0f / (float) Math.sqrt(sqrX));
    }

    // Rotate
    float[] rotated = rotation.rotate(padded);

    // Pack sign bits into first signByteSize bytes
    int signBytes = signByteSize();
    Arrays.fill(dst, 0, dst.length, (byte) 0);
    for (int d = 0; d < paddedDimension; d++) {
      if (rotated[d] >= 0) {
        int byteIndex = d >> 3;
        int bitIndex = d & 7;
        dst[byteIndex] |= (byte) (1 << bitIndex);
      }
    }

    // Compute absolute values and greedy quantize magnitudes
    float[] magnitudes = new float[paddedDimension];
    for (int d = 0; d < paddedDimension; d++) {
      magnitudes[d] = Math.abs(rotated[d]);
    }
    int[] magIndices = greedyQuantize(magnitudes, bits);

    // Pack magnitude indices after sign bits
    byte[] magPacked = packMagnitudes(magIndices, bits);
    System.arraycopy(magPacked, 0, dst, signBytes, magPacked.length);

    return sqrX;
  }

  @Override
  public float[] decode(byte[] encoded) {
    int required = encodedByteSize();
    if (encoded.length < required) {
      throw new IllegalArgumentException(
          "encoded length " + encoded.length + " < required " + required);
    }
    int signBytes = signByteSize();

    // Unpack sign bits to bipolar vector
    float[] rotated = new float[paddedDimension];
    byte[] magPacked = new byte[magByteSize()];
    System.arraycopy(encoded, signBytes, magPacked, 0, magPacked.length);
    int[] magIndices = unpackMagnitudes(magPacked, paddedDimension, bits);
    int maxLevel = (1 << bits) - 1;

    for (int d = 0; d < paddedDimension; d++) {
      int byteIndex = d >> 3;
      int bitIndex = d & 7;
      boolean positive = ((encoded[byteIndex] >> bitIndex) & 1) == 1;
      // Map the quantized index back to [0, 1] relative to the greedy quantize scale.
      float magnitude = (magIndices[d] + 0.5f) / (maxLevel + 0.5f);
      // Heuristic: divide by sqrt(D) to approximate the true coordinate magnitude.
      // For a D-dim unit vector, the expected (RMS) per-coordinate magnitude is 1/sqrt(D),
      // so this gives the correct order of magnitude even though individual coordinates vary
      // widely. Relative magnitudes from greedy quantization are preserved; only the absolute
      // scale is approximate. Use correction-factor scoring for precise distance estimates.
      rotated[d] = (positive ? magnitude : -magnitude) / (float) Math.sqrt(paddedDimension);
    }

    // Inverse rotate
    float[] unrotated = rotation.inverseRotate(rotated);

    // Truncate to original dimension and add centroid
    float[] result = Arrays.copyOf(unrotated, dimension);
    VectorUtil.addInPlace(result, centroid);
    return result;
  }

  @Override
  public ExtendedRaBitQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    int numLongs = numLongs();
    long[][] signCodes = new long[count][numLongs];
    byte[][] magCodes = new byte[count][];
    float[][] corrections = new float[count][NUM_CORRECTIONS];

    byte[] signBuf = new byte[signByteSize()];
    float sqrtPaddedDim = (float) Math.sqrt(paddedDimension);

    for (int i = 0; i < count; i++) {
      float[] vector = dataset.getVector(i);

      // Center and pad
      float[] padded = centerAndPad(vector);

      // sqrX = ||v - centroid||²
      float sqrX = VectorUtil.dotProduct(padded, padded);
      float sqrtSqrX = (float) Math.sqrt(sqrX);

      // Normalize in-place (padded is a fresh allocation; safe to mutate)
      if (sqrX > 0) {
        VectorUtil.scale(padded, 1.0f / sqrtSqrX);
      } else {
        Arrays.fill(padded, 0f);
      }

      // Rotate
      float[] rotated = rotation.rotate(padded);

      // Sign-bit encode
      RaBitQuantizer.packSignBits(rotated, signBuf);
      BinaryQuantizer.bytesToLongs(signBuf, signCodes[i]);

      // Compute absolute values and greedy quantize magnitudes
      float[] magnitudes = new float[paddedDimension];
      for (int d = 0; d < paddedDimension; d++) {
        magnitudes[d] = Math.abs(rotated[d]);
      }
      int[] magIndices = greedyQuantize(magnitudes, bits);
      magCodes[i] = packMagnitudes(magIndices, bits);

      // Compute multi-bit correction factors.
      // The multi-bit code for dimension d is: signed_code[d] = (2*sign[d]-1) * (mag[d]+0.5)
      // x0_multi = cosine(|rotated|, mag+0.5) — the multi-bit quality factor (> 1-bit x0)
      // factorIp_multi = -2*sqrt(sqrX) / (x0_multi * norm_code)
      // where norm_code = sqrt(sum((mag[d]+0.5)²))

      // Also compute 1-bit x0 for the IDX_X0 correction
      float x0 = computeX0(rotated, signBuf, sqrtPaddedDim);

      float normCodeSq = 0f;
      float ipRotCode = 0f;
      float signedCodeSum = 0f; // A = sum((2*sign-1)*(mag+0.5))
      for (int d = 0; d < paddedDimension; d++) {
        float magVal = magIndices[d] + 0.5f;
        normCodeSq += magVal * magVal;
        ipRotCode += magnitudes[d] * magVal; // |rot[d]| * (mag[d]+0.5)
        int longIdx = d >> 6;
        int bitIdx = d & 63;
        boolean positive = ((signCodes[i][longIdx] >> bitIdx) & 1) == 1;
        signedCodeSum += positive ? magVal : -magVal;
      }
      float normCode = (float) Math.sqrt(normCodeSq);
      float x0Multi = (normCode > 0f) ? ipRotCode / normCode : 0f;

      corrections[i][IDX_SQR_X] = sqrX;
      corrections[i][IDX_X0] = x0;

      if (x0Multi > 0f && normCode > 0f) {
        float factorIpMulti = -2.0f * sqrtSqrX / (x0Multi * normCode);
        float factorPpcMulti = factorIpMulti * signedCodeSum;

        corrections[i][IDX_FACTOR_PPC] = factorPpcMulti;
        corrections[i][IDX_FACTOR_IP] = factorIpMulti;

        // Error bound using multi-bit quality
        float x0MultiSq = x0Multi * x0Multi;
        float errorVariance = (1.0f / x0MultiSq - 1.0f) / (paddedDimension - 1);
        corrections[i][IDX_ERROR_FACTOR] =
            ERROR_CONFIDENCE * (float) Math.sqrt(Math.max(errorVariance, 0f)) * 2.0f * sqrtSqrX;
        corrections[i][IDX_XIP_NORM] = 1.0f / (x0Multi * normCode);
      }
    }

    return new ExtendedRaBitQuantizedVectors(this, signCodes, magCodes, corrections, dimension);
  }

  @Override
  public float compressionRatio() {
    int compressedBytes = encodedByteSize() + NUM_CORRECTIONS * Float.BYTES;
    return (dimension * (float) Float.BYTES) / compressedBytes;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors ---

  /** Returns the padded dimension (rounded up to the next multiple of 64). */
  int paddedDimension() {
    return paddedDimension;
  }

  /** Returns the magnitude bit-width. */
  int bits() {
    return bits;
  }

  /** Returns the dataset centroid. */
  float[] centroid() {
    return centroid;
  }

  /** Returns the precomputed padded centroid (zero-padded to paddedDimension). */
  float[] paddedCentroid() {
    return paddedCentroid;
  }

  /** Returns the rotation strategy. */
  Rotation rotation() {
    return rotation;
  }

  /** Returns the number of bytes for the sign-bit portion. */
  int signByteSize() {
    return paddedDimension >> 3;
  }

  /** Returns the number of bytes for the magnitude-code portion. */
  int magByteSize() {
    return (paddedDimension * bits + 7) / 8;
  }

  /** Returns the total encoded byte size (sign bits + magnitude codes). */
  int encodedByteSize() {
    return signByteSize() + magByteSize();
  }

  /** Returns the number of longs needed to store the sign-bit codes. */
  int numLongs() {
    return paddedDimension >> 6;
  }

  // --- Static helpers ---

  /**
   * Greedy rescale quantization: finds the rescale factor t that maximizes cosine similarity
   * between the quantized codes and the input absolute magnitudes.
   *
   * <p>For each candidate t, computes: {@code mag[d] = clamp(round(t * magnitudes[d]), 0,
   * maxLevel)} and picks the t maximizing cosine(quantized + 0.5, magnitudes).
   *
   * @param magnitudes absolute values of rotated coordinates (all non-negative)
   * @param bits bit-width for magnitude quantization
   * @return quantized magnitude indices in [0, maxLevel]
   */
  static int[] greedyQuantize(float[] magnitudes, int bits) {
    int dim = magnitudes.length;
    int maxLevel = (1 << bits) - 1;

    // Find max magnitude for search range
    float maxMag = 0f;
    for (float m : magnitudes) {
      if (m > maxMag) maxMag = m;
    }

    if (maxMag == 0f) {
      return new int[dim];
    }

    // Search range for rescale factor t
    float tMin = 2.0f * maxLevel / (3.0f * maxMag);
    float tMax = (maxLevel + 0.5f) / maxMag;
    float tStep = (tMax - tMin) / NUM_RESCALE_CANDIDATES;

    // normB = sum(magnitudes[d]²) is invariant across all rescale candidates; hoist it out
    // of the loop to avoid dim × NUM_RESCALE_CANDIDATES redundant multiply-accumulates.
    float normB = 0f;
    for (float m : magnitudes) normB += m * m;

    float bestCosine = Float.NEGATIVE_INFINITY;
    float bestT = tMin;

    for (int c = 0; c <= NUM_RESCALE_CANDIDATES; c++) {
      float t = tMin + c * tStep;
      // Compute cosine between (round(t*mag) + 0.5) and mag
      float dot = 0f, normA = 0f;
      for (int d = 0; d < dim; d++) {
        int q = Math.round(t * magnitudes[d]);
        q = Math.max(0, Math.min(maxLevel, q));
        float qVal = q + 0.5f;
        dot += qVal * magnitudes[d];
        normA += qVal * qVal;
      }
      float cosine = (normA > 0 && normB > 0) ? dot / ((float) Math.sqrt(normA * normB)) : 0f;
      if (cosine > bestCosine) {
        bestCosine = cosine;
        bestT = t;
      }
    }

    // Quantize with the best rescale factor
    int[] indices = new int[dim];
    for (int d = 0; d < dim; d++) {
      int q = Math.round(bestT * magnitudes[d]);
      indices[d] = Math.max(0, Math.min(maxLevel, q));
    }
    return indices;
  }

  /**
   * Packs magnitude indices into a byte array using variable-width bit-packing (LSB-first). Zero
   * initializes the buffer before OR-packing.
   *
   * @param indices magnitude indices (one per dimension)
   * @param bits bit-width per index
   * @return packed byte array
   */
  static byte[] packMagnitudes(int[] indices, int bits) {
    int totalBits = indices.length * bits;
    byte[] packed = new byte[(totalBits + 7) / 8];
    int bitPos = 0;
    for (int idx : indices) {
      int byteIdx = bitPos / 8;
      int bitOffset = bitPos % 8;

      if (bitOffset + bits <= 8) {
        packed[byteIdx] |= (byte) (idx << bitOffset);
      } else {
        packed[byteIdx] |= (byte) (idx << bitOffset);
        packed[byteIdx + 1] |= (byte) (idx >> (8 - bitOffset));
      }
      bitPos += bits;
    }
    return packed;
  }

  /**
   * Unpacks magnitude indices from a byte array.
   *
   * @param packed the packed byte array
   * @param dim number of indices to unpack
   * @param bits bit-width per index
   * @return magnitude indices
   */
  static int[] unpackMagnitudes(byte[] packed, int dim, int bits) {
    int mask = (1 << bits) - 1;
    int[] indices = new int[dim];
    int bitPos = 0;
    for (int d = 0; d < dim; d++) {
      int byteIdx = bitPos / 8;
      int bitOffset = bitPos % 8;

      int idx;
      if (bitOffset + bits <= 8) {
        idx = ((packed[byteIdx] & 0xFF) >> bitOffset) & mask;
      } else {
        int lo = (packed[byteIdx] & 0xFF) >>> bitOffset;
        int hi = (packed[byteIdx + 1] & 0xFF) << (8 - bitOffset);
        idx = (lo | hi) & mask;
      }
      indices[d] = idx;
      bitPos += bits;
    }
    return indices;
  }

  // --- Internal helpers ---

  /**
   * Centers and pads a vector: subtracts centroid and zero-pads to paddedDimension.
   *
   * @return a newly allocated padded array (1 allocation)
   */
  private float[] centerAndPad(float[] vector) {
    float[] padded = new float[paddedDimension];
    System.arraycopy(vector, 0, padded, 0, dimension);
    VectorUtil.subInPlace(padded, paddedCentroid);
    return padded;
  }

  /**
   * Computes x0: the inner product between the unit rotated residual and its bipolar quantized
   * form, normalized by sqrt(paddedDim). Same as {@link RaBitQuantizer}'s x0.
   */
  private static float computeX0(float[] rotated, byte[] packed, float sqrtPaddedDim) {
    float dot = 0f;
    for (int d = 0; d < rotated.length; d++) {
      int byteIndex = d >> 3;
      int bitIndex = d & 7;
      dot += ((packed[byteIndex] >> bitIndex) & 1) == 1 ? rotated[d] : -rotated[d];
    }
    return dot / sqrtPaddedDim;
  }
}
