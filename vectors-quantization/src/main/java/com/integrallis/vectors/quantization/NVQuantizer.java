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
 * NVQ (Non-uniform Vector Quantization) quantizer. Learns a per-vector nonlinear transform that
 * adapts to each vector's value distribution before uniform uint8 quantization, achieving better
 * reconstruction accuracy than linear scalar quantization at the same compression ratio.
 *
 * <p><b>Key advantage over ScalarQuantizer:</b> NVQ uses a per-subvector logistic (sigmoid)
 * transform that concentrates quantization levels where values are dense, reducing MSE by up to 3x
 * compared to uniform int8. The nonlinear transform parameters (alpha, x0) are optimized per
 * subvector via grid search.
 *
 * <p><b>Encoding pipeline:</b>
 *
 * <ol>
 *   <li>Center: subtract global mean
 *   <li>Split into M subvectors
 *   <li>Per subvector: find min/max, optimize transform, apply scaled logistic, quantize to uint8
 *   <li>Store: bytes[D] + per-subvector metadata[M * 4 floats]
 * </ol>
 *
 * <p><b>Storage format per vector:</b> D bytes (quantized) + M * 16 bytes (metadata per subvector:
 * alpha, x0, min, max).
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * NVQuantizer nvq = NVQuantizer.train(data, 2);  // 2 subvectors
 * NVQuantizedVectors compressed = nvq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
 * }</pre>
 *
 * @see NQTransform
 * @see NVQuantizedVectors
 */
public final class NVQuantizer implements Quantizer<NVQuantizedVectors> {

  /** Number of floats per subvector metadata: alpha, x0, min, max. */
  static final int METADATA_FLOATS_PER_SV = 4;

  private final int dimension;
  private final int numSubvectors;
  private final int[] subvectorSizes;
  private final float[] globalMean;

  private NVQuantizer(int dimension, int numSubvectors, int[] subvectorSizes, float[] globalMean) {
    this.dimension = dimension;
    this.numSubvectors = numSubvectors;
    this.subvectorSizes = subvectorSizes;
    this.globalMean = globalMean;
  }

  // --- Factory methods ---

  /**
   * Reconstructs a trained {@code NVQuantizer} from previously serialized state. Used by
   * deserialization codecs to restore a quantizer without re-computing the centroid.
   *
   * @param dimension the original vector dimension
   * @param numSubvectors number of subvectors (M)
   * @param subvectorSizes dimensions per subvector
   * @param globalMean the global mean (centroid)
   * @return a reconstructed NVQ quantizer
   */
  public static NVQuantizer fromState(
      int dimension, int numSubvectors, int[] subvectorSizes, float[] globalMean) {
    return new NVQuantizer(dimension, numSubvectors, subvectorSizes, globalMean);
  }

  /**
   * Trains an NVQ quantizer on the given dataset.
   *
   * <p>Training computes the global mean (centroid) and determines the subvector partition. The
   * per-vector transform parameters (alpha, x0) are found lazily during {@link #encode}, not during
   * training.
   *
   * @param dataset the training data
   * @param numSubvectors number of subvectors (M); must be {@code >= 1} and {@code <= dimension}
   * @return a trained NVQ quantizer
   * @throws IllegalArgumentException if numSubvectors is invalid
   */
  public static NVQuantizer train(VectorDataset dataset, int numSubvectors) {
    int dim = dataset.dimension();
    if (numSubvectors < 1 || numSubvectors > dim) {
      throw new IllegalArgumentException(
          "numSubvectors must be in [1, " + dim + "], got " + numSubvectors);
    }

    int[] sizes = computeSubvectorSizes(dim, numSubvectors);
    float[] globalMean = dataset.computeCentroid();
    return new NVQuantizer(dim, numSubvectors, sizes, globalMean);
  }

  /**
   * Trains an NVQ quantizer with a default number of subvectors (roughly 64 dims per subvector).
   *
   * @param dataset the training data
   * @return a trained NVQ quantizer
   */
  public static NVQuantizer train(VectorDataset dataset) {
    int dim = dataset.dimension();
    int m = Math.max(1, dim / 64);
    return train(dataset, m);
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
   * <p>Layout: [quantized bytes: D] [per-subvector metadata: M * 16 bytes (4 floats each)].
   *
   * <p>The encoding optimizes the NQT transform parameters per subvector via grid search, so
   * encoding is more expensive than decoding/scoring.
   *
   * @param vector the input float vector (must have length == dimension)
   * @param dst the destination byte array (must have length >= encodedByteSize())
   * @return 0.0f (metadata is embedded in dst, not returned as a single correction)
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

    // Center
    float[] centered = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      centered[d] = vector[d] - globalMean[d];
    }

    // Encode per subvector
    float[] metadata = new float[numSubvectors * METADATA_FLOATS_PER_SV];
    encodeSubvectors(centered, dst, metadata);

    // Embed metadata into dst after quantized bytes
    int metaOffset = dimension;
    for (int m = 0; m < numSubvectors; m++) {
      int metaIdx = m * METADATA_FLOATS_PER_SV;
      writeFloat(dst, metaOffset, metadata[metaIdx]);
      writeFloat(dst, metaOffset + 4, metadata[metaIdx + 1]);
      writeFloat(dst, metaOffset + 8, metadata[metaIdx + 2]);
      writeFloat(dst, metaOffset + 12, metadata[metaIdx + 3]);
      metaOffset += METADATA_FLOATS_PER_SV * Float.BYTES;
    }

    return 0.0f; // No single correction factor; metadata is embedded
  }

  @Override
  public float[] decode(byte[] encoded) {
    int required = encodedByteSize();
    if (encoded.length < required) {
      throw new IllegalArgumentException(
          "encoded length " + encoded.length + " < required " + required);
    }
    float[] result = new float[dimension];
    float[] p =
        new float[5]; // reusable params: [scaledAlpha, x0, logMin, logisticScale, invScaledAlpha]
    int dimOffset = 0;
    int metaOffset = dimension;

    for (int m = 0; m < numSubvectors; m++) {
      int svDim = subvectorSizes[m];
      float alpha = readFloat(encoded, metaOffset);
      float x0 = readFloat(encoded, metaOffset + 4);
      float min = readFloat(encoded, metaOffset + 8);
      float max = readFloat(encoded, metaOffset + 12);
      loadInvTransformParams(alpha, x0, min, max, p);

      for (int d = 0; d < svDim; d++) {
        int q = encoded[dimOffset + d] & 0xFF;
        float dequantized = NQTransform.scaledLogit(q / 255f, p[4], p[1], p[3], p[2]);
        result[dimOffset + d] = dequantized + globalMean[dimOffset + d];
      }

      dimOffset += svDim;
      metaOffset += METADATA_FLOATS_PER_SV * Float.BYTES;
    }
    return result;
  }

  @Override
  public NVQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    byte[][] quantizedBytes = new byte[count][dimension];
    float[][] subvectorMetadata = new float[count][numSubvectors * METADATA_FLOATS_PER_SV];

    for (int i = 0; i < count; i++) {
      float[] vector = dataset.getVector(i);

      // Center
      float[] centered = new float[dimension];
      for (int d = 0; d < dimension; d++) {
        centered[d] = vector[d] - globalMean[d];
      }

      // Encode per subvector using shared helper
      encodeSubvectors(centered, quantizedBytes[i], subvectorMetadata[i]);
    }

    return new NVQuantizedVectors(this, quantizedBytes, subvectorMetadata, dimension);
  }

  @Override
  public float compressionRatio() {
    // Original: dimension * 4 bytes (float32)
    // Compressed: dimension bytes + numSubvectors * 16 bytes (metadata)
    int compressedBytes = dimension + numSubvectors * METADATA_FLOATS_PER_SV * Float.BYTES;
    return (dimension * (float) Float.BYTES) / compressedBytes;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors ---

  /** Returns the number of subvectors. */
  public int numSubvectors() {
    return numSubvectors;
  }

  /** Returns the dimensions per subvector. */
  public int[] subvectorSizes() {
    return subvectorSizes;
  }

  /** Returns the global mean (centroid). */
  public float[] globalMean() {
    return globalMean;
  }

  /** Returns the total encoded byte size per vector (quantized bytes + metadata). */
  int encodedByteSize() {
    return dimension + numSubvectors * METADATA_FLOATS_PER_SV * Float.BYTES;
  }

  // --- Inverse transform helpers (package-private for use by NVQuantizedVectors) ---

  /**
   * Computes inverse NQT transform parameters for a single subvector into a pre-allocated array,
   * eliminating the repeated 6-line setup in scoring and decode paths.
   *
   * <p>Output slots: {@code [0]} scaledAlpha, {@code [1]} x0, {@code [2]} logMin (logistic bias),
   * {@code [3]} logisticScale, {@code [4]} inverseScaledAlpha.
   *
   * @param out pre-allocated {@code float[5]} array; mutated in place
   */
  static void loadInvTransformParams(float alpha, float x0, float min, float max, float[] out) {
    float range = max - min;
    out[0] = (range > 1e-10f) ? alpha / range : alpha;
    float logMin = NQTransform.logistic(min, out[0], x0);
    float logMax = NQTransform.logistic(max, out[0], x0);
    float logRange = logMax - logMin;
    out[1] = x0;
    out[2] = logMin;
    out[3] = (logRange > 1e-10f) ? 1f / logRange : 1f;
    out[4] = (alpha > 1e-10f) ? range / alpha : range;
  }

  /**
   * Convenience overload reading {@code alpha, x0, min, max} from a metadata array at {@code
   * metaIdx}.
   */
  static void loadInvTransformParams(float[] metadata, int metaIdx, float[] out) {
    loadInvTransformParams(
        metadata[metaIdx],
        metadata[metaIdx + 1],
        metadata[metaIdx + 2],
        metadata[metaIdx + 3],
        out);
  }

  // --- Internal helpers ---

  /**
   * Encodes the centered vector into quantized bytes and per-subvector metadata. This is the shared
   * encoding pipeline used by both {@link #encode(float[], byte[])} and {@link
   * #encodeAll(VectorDataset)}.
   *
   * @param centered the centered vector (v - globalMean)
   * @param quantizedOut destination for quantized uint8 bytes (must have length >= dimension)
   * @param metadataOut destination for per-subvector metadata [alpha, x0, min, max] per subvector
   */
  private void encodeSubvectors(float[] centered, byte[] quantizedOut, float[] metadataOut) {
    int dimOffset = 0;
    for (int m = 0; m < numSubvectors; m++) {
      int svDim = subvectorSizes[m];
      int metaIdx = m * METADATA_FLOATS_PER_SV;

      // Find min/max
      float min = Float.POSITIVE_INFINITY;
      float max = Float.NEGATIVE_INFINITY;
      for (int d = 0; d < svDim; d++) {
        float v = centered[dimOffset + d];
        if (v < min) min = v;
        if (v > max) max = v;
      }

      // Expand degenerate ranges
      if (max - min < 1e-10f) {
        float mid = (min + max) / 2f;
        min = mid - 0.5f;
        max = mid + 0.5f;
      }

      // Optimize transform parameters
      float[] sv = new float[svDim];
      System.arraycopy(centered, dimOffset, sv, 0, svDim);
      float[] params = NQTransform.optimizeTransform(sv, min, max);
      float alpha = params[0];
      float x0 = params[1];

      // Precompute scaling constants
      float range = max - min;
      float scaledAlpha = alpha / range;
      float logMin = NQTransform.logistic(min, scaledAlpha, x0);
      float logMax = NQTransform.logistic(max, scaledAlpha, x0);
      float logRange = logMax - logMin;
      float logisticScale = (logRange > 1e-10f) ? 1f / logRange : 1f;

      // Quantize to uint8
      for (int d = 0; d < svDim; d++) {
        float mapped =
            NQTransform.scaledLogistic(
                centered[dimOffset + d], scaledAlpha, x0, logMin, logisticScale);
        int q = Math.round(mapped * 255f);
        q = Math.max(0, Math.min(255, q));
        quantizedOut[dimOffset + d] = (byte) q;
      }

      // Store metadata
      metadataOut[metaIdx] = alpha;
      metadataOut[metaIdx + 1] = x0;
      metadataOut[metaIdx + 2] = min;
      metadataOut[metaIdx + 3] = max;

      dimOffset += svDim;
    }
  }

  /**
   * Computes subvector sizes for the given dimension and number of subvectors. Distributes
   * dimensions as evenly as possible; the first {@code dimension % numSubvectors} subvectors each
   * get one extra dimension.
   */
  static int[] computeSubvectorSizes(int dimension, int numSubvectors) {
    int[] sizes = new int[numSubvectors];
    int base = dimension / numSubvectors;
    int remainder = dimension % numSubvectors;
    for (int m = 0; m < numSubvectors; m++) {
      sizes[m] = base + (m < remainder ? 1 : 0);
    }
    return sizes;
  }

  /** Writes a float to a byte array in little-endian order at the given offset. */
  private static void writeFloat(byte[] dst, int offset, float value) {
    int bits = Float.floatToRawIntBits(value);
    dst[offset] = (byte) bits;
    dst[offset + 1] = (byte) (bits >> 8);
    dst[offset + 2] = (byte) (bits >> 16);
    dst[offset + 3] = (byte) (bits >> 24);
  }

  /** Reads a float from a byte array in little-endian order at the given offset. */
  private static float readFloat(byte[] src, int offset) {
    int bits =
        (src[offset] & 0xFF)
            | ((src[offset + 1] & 0xFF) << 8)
            | ((src[offset + 2] & 0xFF) << 16)
            | ((src[offset + 3] & 0xFF) << 24);
    return Float.intBitsToFloat(bits);
  }
}
