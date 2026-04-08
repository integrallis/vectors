package com.integrallis.vectors.quantization;

import java.util.Arrays;
import java.util.Random;

/**
 * Scalar quantization of float32 vectors to int8 (7-bit, [0, 127]) or int4 (4-bit, [0, 15]).
 *
 * <p>Follows the Lucene approach:
 *
 * <ul>
 *   <li>Quantize: {@code byte = round((value - minQ) * maxVal / (maxQ - minQ))}
 *   <li>Dequantize: {@code float = byte * (maxQ - minQ) / maxVal + minQ}
 *   <li>Per-vector correction factor for scoring accuracy
 *   <li>Confidence-interval-based quantile selection to handle outliers
 * </ul>
 *
 * <p>Int4 mode packs two 4-bit values per byte (low nibble = even index, high nibble = odd index),
 * achieving 8x compression vs float32.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * ScalarQuantizer sq = ScalarQuantizer.train(data);           // int8 (default)
 * ScalarQuantizer sq4 = ScalarQuantizer.train(data, ScalarBits.INT4);  // int4
 * ScalarQuantizedVectors compressed = sq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
 * }</pre>
 */
public final class ScalarQuantizer implements Quantizer<ScalarQuantizedVectors> {

  /** Maximum number of vectors sampled for quantile computation. */
  static final int SCALAR_QUANTIZATION_SAMPLE_SIZE = 25_000;

  private final int dimension;
  private final ScalarBits bits;
  private final float minQuantile;
  private final float maxQuantile;
  private final float scale; // maxVal / (maxQ - minQ)
  private final float alpha; // (maxQ - minQ) / maxVal
  private final float constMultiplier; // alpha * alpha

  private ScalarQuantizer(int dimension, ScalarBits bits, float minQuantile, float maxQuantile) {
    this.dimension = dimension;
    this.bits = bits;
    this.minQuantile = minQuantile;
    this.maxQuantile = maxQuantile;
    float range = maxQuantile - minQuantile;
    int maxVal = bits.maxValue();
    this.scale = (range == 0f) ? 0f : maxVal / range;
    this.alpha = (range == 0f) ? 0f : range / maxVal;
    this.constMultiplier = alpha * alpha;
  }

  // --- Factory methods ---

  /**
   * Trains an int8 scalar quantizer with default confidence interval (0.99).
   *
   * @param dataset the training data
   * @return a trained int8 scalar quantizer
   */
  public static ScalarQuantizer train(VectorDataset dataset) {
    return train(dataset, 0.99f);
  }

  /**
   * Trains an int8 scalar quantizer with the given confidence interval.
   *
   * @param dataset the training data
   * @param confidenceInterval (0.0, 1.0]; 1.0 uses absolute min/max
   * @return a trained int8 scalar quantizer
   */
  public static ScalarQuantizer train(VectorDataset dataset, float confidenceInterval) {
    return train(dataset, ScalarBits.INT8, confidenceInterval);
  }

  /**
   * Trains a scalar quantizer with the specified bit width and default confidence interval (0.99).
   *
   * @param dataset the training data
   * @param bits the quantization bit width
   * @return a trained scalar quantizer
   */
  public static ScalarQuantizer train(VectorDataset dataset, ScalarBits bits) {
    return train(dataset, bits, 0.99f);
  }

  /**
   * Trains a scalar quantizer with the specified bit width and confidence interval.
   *
   * @param dataset the training data
   * @param bits the quantization bit width
   * @param confidenceInterval (0.0, 1.0]; 1.0 uses absolute min/max
   * @return a trained scalar quantizer
   */
  public static ScalarQuantizer train(
      VectorDataset dataset, ScalarBits bits, float confidenceInterval) {
    if (confidenceInterval <= 0f || confidenceInterval > 1f) {
      throw new IllegalArgumentException(
          "Confidence interval must be in (0, 1]: " + confidenceInterval);
    }
    float[] quantiles = computeQuantiles(dataset, confidenceInterval);
    return new ScalarQuantizer(dataset.dimension(), bits, quantiles[0], quantiles[1]);
  }

  /**
   * Creates an int8 scalar quantizer with explicit quantile bounds.
   *
   * @param dimension the vector dimensionality
   * @param minQuantile the lower quantile bound
   * @param maxQuantile the upper quantile bound
   * @return a scalar quantizer with the given bounds
   */
  public static ScalarQuantizer fromQuantiles(int dimension, float minQuantile, float maxQuantile) {
    return fromQuantiles(dimension, ScalarBits.INT8, minQuantile, maxQuantile);
  }

  /**
   * Creates a scalar quantizer with explicit quantile bounds and bit width.
   *
   * @param dimension the vector dimensionality
   * @param bits the quantization bit width
   * @param minQuantile the lower quantile bound
   * @param maxQuantile the upper quantile bound
   * @return a scalar quantizer with the given bounds
   */
  public static ScalarQuantizer fromQuantiles(
      int dimension, ScalarBits bits, float minQuantile, float maxQuantile) {
    if (minQuantile >= maxQuantile) {
      throw new IllegalArgumentException(
          "minQuantile must be < maxQuantile: " + minQuantile + " >= " + maxQuantile);
    }
    return new ScalarQuantizer(dimension, bits, minQuantile, maxQuantile);
  }

  // --- Quantizer interface ---

  @Override
  public byte[] encode(float[] vector) {
    byte[] dst = new byte[bits.encodedByteSize(vector.length)];
    encode(vector, dst);
    return dst;
  }

  @Override
  public float encode(float[] vector, byte[] dst) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }
    return switch (bits) {
      case INT8 -> encodeInt8(vector, dst);
      case INT4 -> encodeInt4(vector, dst);
    };
  }

  @Override
  public float[] decode(byte[] encoded) {
    return switch (bits) {
      case INT8 -> decodeInt8(encoded);
      case INT4 -> decodeInt4(encoded);
    };
  }

  @Override
  public ScalarQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    int dim = dataset.dimension();
    int encodedSize = bits.encodedByteSize(dim);
    byte[][] quantized = new byte[count][encodedSize];
    float[] corrections = new float[count];
    for (int i = 0; i < count; i++) {
      corrections[i] = encode(dataset.getVector(i), quantized[i]);
    }
    return new ScalarQuantizedVectors(this, quantized, corrections, dim);
  }

  @Override
  public float compressionRatio() {
    return bits.compressionRatio();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors ---

  /** Returns the bit-width configuration. */
  public ScalarBits bits() {
    return bits;
  }

  /** Returns the lower quantile bound. */
  public float minQuantile() {
    return minQuantile;
  }

  /** Returns the upper quantile bound. */
  public float maxQuantile() {
    return maxQuantile;
  }

  /** Returns the dequantization scale factor: {@code (maxQ - minQ) / maxVal}. */
  public float alpha() {
    return alpha;
  }

  /** Returns the quantization scale factor: {@code maxVal / (maxQ - minQ)}. */
  public float scale() {
    return scale;
  }

  /** Returns {@code alpha * alpha}, the constant multiplier for quantized scoring. */
  public float constMultiplier() {
    return constMultiplier;
  }

  // --- Shared per-element quantization helpers ---

  /**
   * Clamps {@code v} to [minQuantile, maxQuantile], scales to [0, maxValue], and rounds. This is
   * the single definition of the quantization formula used by all encode methods.
   */
  private int quantizeValue(float v) {
    float dxc = Math.max(minQuantile, Math.min(maxQuantile, v)) - minQuantile;
    int q = Math.round(scale * dxc);
    return Math.max(0, Math.min(bits.maxValue(), q));
  }

  /**
   * Returns the correction term for one dimension. The correction accumulates the error introduced
   * by quantization so that dot-product scoring can compensate for it.
   *
   * <p>Formula (Lucene derivation): {@code minQ * (v - minQ/2) + (dx - dxq) * dxq}, where {@code dx
   * = v - minQ} and {@code dxq = q * alpha} (the dequantized value minus minQ).
   */
  private float correctionTerm(float v, int q) {
    float dx = v - minQuantile;
    float dxq = q * alpha;
    return minQuantile * (v - minQuantile / 2f) + (dx - dxq) * dxq;
  }

  // --- Int8 encode/decode ---

  private float encodeInt8(float[] vector, byte[] dst) {
    float correction = 0f;
    for (int i = 0; i < vector.length; i++) {
      int q = quantizeValue(vector[i]);
      dst[i] = (byte) q;
      correction += correctionTerm(vector[i], q);
    }
    return correction;
  }

  private float[] decodeInt8(byte[] encoded) {
    float[] result = new float[encoded.length];
    for (int i = 0; i < encoded.length; i++) {
      result[i] = (encoded[i] & 0xFF) * alpha + minQuantile;
    }
    return result;
  }

  // --- Int4 encode/decode ---

  /**
   * Encodes a float vector to packed int4 bytes. Two nibbles per byte: low nibble = even dimension,
   * high nibble = odd dimension.
   *
   * <p>{@code dst} is zero-initialised before packing so it is safe to call with a reused buffer.
   */
  private float encodeInt4(float[] vector, byte[] dst) {
    Arrays.fill(dst, (byte) 0); // zero-init: OR-ing high nibbles requires a clean slate
    float correction = 0f;
    for (int i = 0; i < vector.length; i++) {
      int q = quantizeValue(vector[i]);
      int byteIndex = i >> 1;
      if ((i & 1) == 0) {
        dst[byteIndex] = (byte) (q & 0x0F); // low nibble
      } else {
        dst[byteIndex] |= (byte) ((q & 0x0F) << 4); // high nibble
      }
      correction += correctionTerm(vector[i], q);
    }
    return correction;
  }

  /**
   * Decodes packed int4 bytes back to floats. Low nibble = even dimension, high nibble = odd
   * dimension.
   */
  private float[] decodeInt4(byte[] encoded) {
    float[] result = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      int byteIndex = i >> 1;
      int nibble =
          ((i & 1) == 0)
              ? (encoded[byteIndex] & 0x0F)
              : ((encoded[byteIndex] >> 4) & 0x0F); // & 0x0F cancels sign-extension
      result[i] = nibble * alpha + minQuantile;
    }
    return result;
  }

  /**
   * Unpacks an int4-packed byte array into a full byte array (one value per byte). Used for scoring
   * via existing byte dot-product kernels.
   *
   * @param packed the packed int4 bytes (dim/2 bytes)
   * @param unpacked the destination array (dim bytes, values in [0, 15])
   */
  void unpackInt4(byte[] packed, byte[] unpacked) {
    for (int i = 0; i < dimension; i++) {
      int byteIndex = i >> 1;
      unpacked[i] =
          (byte) (((i & 1) == 0) ? (packed[byteIndex] & 0x0F) : ((packed[byteIndex] >> 4) & 0x0F));
    }
  }

  /**
   * Encodes a vector to unpacked int4 values (one value per byte, each in [0, 15]). Used for query
   * encoding where we need the full byte array for dot product computation.
   */
  float encodeInt4Unpacked(float[] vector, byte[] dst) {
    float correction = 0f;
    for (int i = 0; i < vector.length; i++) {
      int q = quantizeValue(vector[i]);
      dst[i] = (byte) q;
      correction += correctionTerm(vector[i], q);
    }
    return correction;
  }

  // --- Quantile computation ---

  /**
   * Computes [minQuantile, maxQuantile] from the dataset using confidence interval percentiles. For
   * datasets larger than {@link #SCALAR_QUANTIZATION_SAMPLE_SIZE}, uses reservoir sampling via
   * {@link ReservoirSampler} (deterministic seed 42).
   */
  static float[] computeQuantiles(VectorDataset dataset, float confidenceInterval) {
    int dim = dataset.dimension();
    int datasetSize = dataset.size();

    // Sample dataset indices (identity mapping when dataset fits in reservoir)
    int[] indices =
        ReservoirSampler.sampleIndices(
            datasetSize, SCALAR_QUANTIZATION_SAMPLE_SIZE, new Random(42L));
    int sampleSize = indices.length;

    // Collect all scalar values from sampled vectors into a flat array for sorting
    float[] allValues = new float[sampleSize * dim];
    for (int i = 0; i < sampleSize; i++) {
      System.arraycopy(dataset.getVector(indices[i]), 0, allValues, i * dim, dim);
    }

    Arrays.sort(allValues);

    if (confidenceInterval >= 1.0f) {
      float min = allValues[0];
      float max = allValues[allValues.length - 1];
      if (min == max) {
        return expandDegenerateRange(min);
      }
      return new float[] {min, max};
    }

    int selectorIndex = (int) (allValues.length * (1f - confidenceInterval) / 2f + 0.5f);
    selectorIndex = Math.max(0, Math.min(selectorIndex, allValues.length - 1));
    float minQ = allValues[selectorIndex];
    float maxQ = allValues[allValues.length - 1 - selectorIndex];

    if (minQ == maxQ) {
      return expandDegenerateRange(minQ);
    }

    return new float[] {minQ, maxQ};
  }

  /** Expands a degenerate range (min == max) by a small epsilon. */
  private static float[] expandDegenerateRange(float value) {
    float abs = Math.abs(value);
    float eps = abs > 0 ? abs * 1e-6f : 1e-6f;
    return new float[] {value - eps, value + eps};
  }
}
