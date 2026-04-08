package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;

/**
 * RaBitQ (Randomized Binary Quantization) quantizer. Implements the SIGMOD 2024 algorithm that
 * achieves higher recall than simple binary quantization (BQ/BBQ) at the same 32x compression ratio
 * by applying a random orthogonal rotation before sign-bit encoding.
 *
 * <p><b>Encoding pipeline:</b> center → pad → normalize → rotate → sign-bit encode, with 5
 * per-vector correction factors for unbiased distance estimation.
 *
 * <p><b>Key theoretical property:</b> The random rotation maps each normalized, centered residual
 * to the nearest vertex of a hypercube inscribed in the unit sphere, producing an <em>unbiased</em>
 * distance estimator with theoretical error bounds.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * RaBitQuantizer rq = RaBitQuantizer.train(data);
 * RaBitQuantizedVectors compressed = rq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.EUCLIDEAN);
 * }</pre>
 *
 * @see Rotation
 * @see RandomRotation
 * @see RaBitQuantizedVectors
 */
public final class RaBitQuantizer implements Quantizer<RaBitQuantizedVectors> {

  /** Number of per-vector correction floats: sqrX, x0, factorPpc, factorIp, errorFactor. */
  static final int NUM_CORRECTIONS = 5;

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

  /** Confidence multiplier for error bound (≈ 1.9σ covers ~97% of estimation errors). */
  private static final float ERROR_CONFIDENCE = 1.9f;

  private final int dimension;
  private final int paddedDimension;
  private final float[] centroid;
  private final float[] paddedCentroid; // precomputed padded centroid for SIMD centerAndPad
  private final Rotation rotation;

  private RaBitQuantizer(int dimension, int paddedDimension, float[] centroid, Rotation rotation) {
    this.dimension = dimension;
    this.paddedDimension = paddedDimension;
    this.centroid = centroid;
    this.paddedCentroid = new float[paddedDimension];
    System.arraycopy(centroid, 0, this.paddedCentroid, 0, dimension);
    this.rotation = rotation;
  }

  // --- Factory methods ---

  /**
   * Trains a RaBitQ quantizer on the given dataset with default seed (42L).
   *
   * @param dataset the training data
   * @return a trained RaBitQ quantizer
   */
  public static RaBitQuantizer train(VectorDataset dataset) {
    return train(dataset, 42L);
  }

  /**
   * Trains a RaBitQ quantizer on the given dataset using a dense random rotation.
   *
   * <p>Training computes the dataset centroid and generates a random orthogonal rotation matrix.
   * The dimension is padded to the next multiple of 64 for efficient bit-packing.
   *
   * @param dataset the training data
   * @param seed random seed for the rotation matrix (deterministic for reproducibility)
   * @return a trained RaBitQ quantizer
   */
  public static RaBitQuantizer train(VectorDataset dataset, long seed) {
    int dim = dataset.dimension();
    int paddedDim = ((dim + 63) / 64) * 64;
    Rotation rotation = RandomRotation.generate(paddedDim, seed);
    return train(dataset, rotation);
  }

  /**
   * Trains a RaBitQ quantizer on the given dataset with a pluggable rotation strategy.
   *
   * <p>The rotation's dimension must match the padded dimension (next multiple of 64 above the
   * dataset dimension). Use this overload to plug in alternative rotation strategies such as {@link
   * GivensRotation} or {@link QuaternionRotation} for faster encoding.
   *
   * @param dataset the training data
   * @param rotation the rotation strategy (must have dimension == padded dimension)
   * @return a trained RaBitQ quantizer
   * @throws IllegalArgumentException if the rotation dimension doesn't match the padded dimension
   */
  public static RaBitQuantizer train(VectorDataset dataset, Rotation rotation) {
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
    return new RaBitQuantizer(dim, paddedDim, centroid, rotation);
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
   * <p>Pipeline: center → pad → compute norm² → normalize → rotate → sign-bit pack.
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

    // Center and pad
    float[] padded = centerAndPad(vector);

    // Compute ||v - centroid||² before normalization (uses SIMD dot product)
    float sqrX = VectorUtil.dotProduct(padded, padded);

    // Normalize to unit vector
    if (sqrX > 0) {
      float invNorm = 1.0f / (float) Math.sqrt(sqrX);
      VectorUtil.scale(padded, invNorm);
    }

    // Rotate
    float[] rotated = rotation.rotate(padded);

    // Sign-bit encode into dst
    packSignBits(rotated, dst);

    return sqrX;
  }

  @Override
  public float[] decode(byte[] encoded) {
    // Unpack bits to {-1, +1} vector in padded space
    float[] bipolar = new float[paddedDimension];
    for (int d = 0; d < paddedDimension; d++) {
      int byteIndex = d >> 3;
      int bitIndex = d & 7;
      boolean bit = ((encoded[byteIndex] >> bitIndex) & 1) == 1;
      bipolar[d] = bit ? 1.0f : -1.0f;
    }

    // Scale to unit hypercube vertex: 1/sqrt(paddedDim)
    float scale = 1.0f / (float) Math.sqrt(paddedDimension);
    VectorUtil.scale(bipolar, scale);

    // Inverse rotate
    float[] unrotated = rotation.inverseRotate(bipolar);

    // Truncate to original dimension and add centroid
    float[] result = Arrays.copyOf(unrotated, dimension);
    VectorUtil.addInPlace(result, centroid);
    return result;
  }

  @Override
  public RaBitQuantizedVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    int numLongs = numLongs();
    long[][] codes = new long[count][numLongs];
    float[][] corrections = new float[count][NUM_CORRECTIONS];

    byte[] buf = new byte[encodedByteSize()];
    float sqrtPaddedDim = (float) Math.sqrt(paddedDimension);

    for (int i = 0; i < count; i++) {
      float[] vector = dataset.getVector(i);

      // Center and pad
      float[] padded = centerAndPad(vector);

      // sqrX = ||v - centroid||² (SIMD)
      float sqrX = VectorUtil.dotProduct(padded, padded);
      float sqrtSqrX = (float) Math.sqrt(sqrX);

      // Normalize in-place (padded is a fresh allocation from centerAndPad; safe to mutate).
      // This mirrors encode(float[], byte[]) and avoids an extra Arrays.copyOf per vector.
      if (sqrX > 0) {
        VectorUtil.scale(padded, 1.0f / sqrtSqrX);
      } else {
        // Zero-norm vector: leave as zero; all corrections stay 0 below.
        Arrays.fill(padded, 0f);
      }

      // Rotate (SIMD dot products per row)
      float[] rotated = rotation.rotate(padded);

      // Sign-bit encode
      packSignBits(rotated, buf);
      BinaryQuantizer.bytesToLongs(buf, codes[i]);

      // x0 = sum_d(rotated[d] * bipolar[d]) / sqrt(D)  where bipolar[d] = sign(rotated[d]).
      // Because rotated[d] * sign(rotated[d]) = |rotated[d]|, this is a scalar sum of absolutes —
      // no allocation needed.
      float x0 = computeX0(rotated, buf, sqrtPaddedDim);

      // Compute correction factors
      int popcount = totalBitCount(codes[i]);
      corrections[i][IDX_SQR_X] = sqrX;
      corrections[i][IDX_X0] = x0;

      if (x0 != 0f) {
        // factorIp = -2 * sqrt(sqrX) / (x0 * sqrt(paddedDim))
        float factorIp = -2.0f * sqrtSqrX / (x0 * sqrtPaddedDim);
        // factorPpc = factorIp * (2 * popcount - paddedDim)
        float factorPpc = factorIp * (2 * popcount - paddedDimension);

        corrections[i][IDX_FACTOR_PPC] = factorPpc;
        corrections[i][IDX_FACTOR_IP] = factorIp;

        // Error bound: 1.9 * sqrt((1/x0² - 1) / (paddedDim - 1)) * 2 * sqrt(sqrX)
        float x0Sq = x0 * x0;
        float errorVariance = (1.0f / x0Sq - 1.0f) / (paddedDimension - 1);
        corrections[i][IDX_ERROR_FACTOR] =
            ERROR_CONFIDENCE * (float) Math.sqrt(Math.max(errorVariance, 0f)) * 2.0f * sqrtSqrX;
      }
      // else: zero-norm vector, all correction factors remain 0
    }

    return new RaBitQuantizedVectors(this, codes, corrections, dimension);
  }

  @Override
  public float compressionRatio() {
    // Original: dimension * 4 bytes (float32)
    // Compressed: paddedDimension/8 bytes (bits) + NUM_CORRECTIONS * 4 bytes (corrections)
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

  /** Returns the dataset centroid (mean vector). */
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

  /** Returns the number of bytes for the bit-code portion of one encoded vector. */
  int encodedByteSize() {
    return paddedDimension >> 3; // paddedDimension is always a multiple of 64, so /8 is exact
  }

  /** Returns the number of longs needed to store one encoded vector. */
  int numLongs() {
    return paddedDimension >> 6; // paddedDimension is always a multiple of 64
  }

  // --- Internal helpers ---

  /**
   * Centers and pads a vector: subtracts centroid and zero-pads to paddedDimension. Uses the
   * precomputed padded centroid for SIMD-accelerated subtraction via {@link VectorUtil#subInPlace}.
   */
  private float[] centerAndPad(float[] vector) {
    float[] padded = new float[paddedDimension];
    System.arraycopy(vector, 0, padded, 0, dimension);
    VectorUtil.subInPlace(padded, paddedCentroid);
    return padded;
  }

  /**
   * Packs sign bits of a float vector into bytes. Zero-initializes the buffer to prevent stale bits
   * from reused buffers (required for OR-packing correctness).
   *
   * @param vector the float vector (bit[d] = 1 if vector[d] >= 0)
   * @param dst the destination byte array
   */
  static void packSignBits(float[] vector, byte[] dst) {
    Arrays.fill(dst, (byte) 0);
    for (int d = 0; d < vector.length; d++) {
      if (vector[d] >= 0) {
        int byteIndex = d >> 3;
        int bitIndex = d & 7;
        dst[byteIndex] |= (byte) (1 << bitIndex);
      }
    }
  }

  /**
   * Computes x0: the inner product between the unit rotated residual and its bipolar quantized
   * form, normalized by sqrt(paddedDim).
   *
   * <p>{@code x0 = dot(rotated, bipolar) / sqrt(paddedDim)} where {@code bipolar[d] =
   * sign(rotated[d])}. Since {@code rotated[d] * sign(rotated[d]) = |rotated[d]|}, this is
   * equivalent to {@code sum(|rotated[d]|) / sqrt(paddedDim)} — computed without allocating a
   * bipolar array.
   *
   * <p>x0 is always in (0, 1] for any non-zero unit vector (by Cauchy-Schwarz) and measures
   * quantization quality: higher x0 means the sign-bit encoding better preserves the direction.
   *
   * @param rotated the rotated unit vector
   * @param packed the sign-bit packed bytes (bit[d] = 1 iff rotated[d] >= 0)
   * @param sqrtPaddedDim precomputed sqrt(paddedDimension)
   * @return the x0 correction factor
   */
  private static float computeX0(float[] rotated, byte[] packed, float sqrtPaddedDim) {
    // rotated[d] * bipolar[d] = rotated[d] * sign(rotated[d]) = |rotated[d]|
    // so dot(rotated, bipolar) = sum |rotated[d]| — no allocation required.
    float dot = 0f;
    for (int d = 0; d < rotated.length; d++) {
      int byteIndex = d >> 3;
      int bitIndex = d & 7;
      dot += ((packed[byteIndex] >> bitIndex) & 1) == 1 ? rotated[d] : -rotated[d];
    }
    return dot / sqrtPaddedDim;
  }

  /** Returns the total number of 1-bits across all longs. */
  private static int totalBitCount(long[] bits) {
    int count = 0;
    for (long l : bits) {
      count += Long.bitCount(l);
    }
    return count;
  }
}
