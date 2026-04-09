package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;

/**
 * Compressed vector storage produced by {@link ExtendedRaBitQuantizer}. Stores 1-bit sign codes
 * plus multi-bit magnitude codes with per-vector correction factors for asymmetric distance
 * estimation (zero allocation per score call).
 *
 * <p><b>Two-layer scoring:</b>
 *
 * <ul>
 *   <li><b>Layer 1 (coarse, sign-only):</b> Uses only the 1-bit sign codes with corrections derived
 *       from x0 and popcount — identical to 1-bit RaBitQ. Cheap to compute (no magnitude
 *       unpacking). Available via {@link #coarseScoreFunctionFor} for early pruning in graph
 *       search.
 *   <li><b>Layer 2 (refined, sign + magnitude):</b> Uses both sign and magnitude codes with
 *       multi-bit correction factors for tighter distance estimates. Available via {@link
 *       #scoreFunctionFor}.
 * </ul>
 *
 * <p><b>Layer 2 formula:</b>
 *
 * <pre>
 * signedMagIp = sum_d(q_byte[d] * (2*sign[d]-1) * mag[d])
 * ip_1bit     = sum_d(q_byte[d] * sign[d])              (reuses 1-bit RaBitQ kernel)
 * B_full      = signedMagIp + 0.5 * (2*ip_1bit - sumQ)  (accounts for +0.5 greedy offset)
 * dist        = sqrX + sqrY + factorPpc*vl + factorIp * widthOver255 * B_full
 * </pre>
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 *
 * @see ExtendedRaBitQuantizer
 */
public final class ExtendedRaBitQuantizedVectors implements CompressedVectors {

  private final ExtendedRaBitQuantizer quantizer;
  private final long[][] signCodes; // [size][numLongs]
  private final byte[][] magCodes; // [size][magByteSize]
  private final float[][] corrections; // [size][6]
  private final int dimension;

  ExtendedRaBitQuantizedVectors(
      ExtendedRaBitQuantizer quantizer,
      long[][] signCodes,
      byte[][] magCodes,
      float[][] corrections,
      int dimension) {
    this.quantizer = quantizer;
    this.signCodes = signCodes;
    this.magCodes = magCodes;
    this.corrections = corrections;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return signCodes.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Pre-computed query state shared by both Layer 1 ({@link #coarseScoreFunctionFor}) and Layer 2
   * ({@link #scoreFunctionFor}) scoring. Centralises the centering → rotation → uint8 quantization
   * pipeline so it is never duplicated.
   */
  private record PreparedQuery(
      byte[] queryBytes,
      float vl,
      float widthOver255,
      int sumQ,
      float sqrY,
      float queryNorm,
      float sqrtPaddedDim) {}

  /**
   * Centers, rotates, and uint8-quantizes the query. Called once per {@code scoreFunctionFor} /
   * {@code coarseScoreFunctionFor} invocation; all results are captured by the returned lambda.
   */
  private PreparedQuery prepareQuery(float[] query) {
    int paddedDim = quantizer.paddedDimension();

    // Center and pad (no normalization — query magnitude is absorbed into vl/width)
    float[] centeredQuery = new float[paddedDim];
    System.arraycopy(query, 0, centeredQuery, 0, query.length);
    VectorUtil.subInPlace(centeredQuery, quantizer.paddedCentroid());

    float sqrY = VectorUtil.dotProduct(centeredQuery, centeredQuery);
    float queryNorm = (float) Math.sqrt(sqrY);

    float[] queryRotated = quantizer.rotation().rotate(centeredQuery);

    // Find min/max for uint8 scaling
    float vl = Float.POSITIVE_INFINITY;
    float vh = Float.NEGATIVE_INFINITY;
    for (float v : queryRotated) {
      if (v < vl) vl = v;
      if (v > vh) vh = v;
    }
    float width = vh - vl;
    float scale = width > 0 ? 255.0f / width : 0f;

    byte[] queryBytes = new byte[paddedDim];
    int sumQ = 0;
    for (int d = 0; d < paddedDim; d++) {
      int q = Math.round((queryRotated[d] - vl) * scale);
      q = Math.max(0, Math.min(255, q));
      queryBytes[d] = (byte) q;
      sumQ += q;
    }

    return new PreparedQuery(
        queryBytes, vl, width / 255.0f, sumQ, sqrY, queryNorm, (float) Math.sqrt(paddedDim));
  }

  @Override
  public ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction) {
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Expected query dimension " + dimension + ", got " + query.length);
    }
    int bits = quantizer.bits();
    PreparedQuery pq = prepareQuery(query);

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal -> {
            float dist =
                estimateL2Multi(
                    signCodes[ordinal],
                    magCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    bits);
            return 1f / (1f + Math.max(dist, 0f));
          };
      case DOT_PRODUCT ->
          ordinal -> {
            float dist =
                estimateL2Multi(
                    signCodes[ordinal],
                    magCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    bits);
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            return Math.max((1f + approxDot) / 2f, 0f);
          };
      case COSINE ->
          ordinal -> {
            float dist =
                estimateL2Multi(
                    signCodes[ordinal],
                    magCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    bits);
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float vecNorm = (float) Math.sqrt(sqrX);
            if (pq.queryNorm() == 0f || vecNorm == 0f) return 0f;
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            float cosine = approxDot / (pq.queryNorm() * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            float dist =
                estimateL2Multi(
                    signCodes[ordinal],
                    magCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    bits);
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            return SimilarityFunction.scaleMaxInnerProductScore(approxDot);
          };
    };
  }

  /**
   * Returns a cheap Layer 1 score function that uses <b>only sign-bit codes</b> (no magnitude
   * unpacking). This produces the same distance estimate as 1-bit RaBitQ — less accurate than the
   * full multi-bit estimate from {@link #scoreFunctionFor}, but cheaper to compute.
   *
   * <p><b>Use case:</b> During graph traversal (HNSW, Vamana), compute the cheap Layer 1 estimate
   * first. Only compute the expensive Layer 2 ({@link #scoreFunctionFor}) for candidates that pass
   * a pruning threshold. This gives the accuracy of multi-bit RaBitQ with the throughput of 1-bit
   * for most candidates.
   *
   * <p><b>Layer 1 formula:</b>
   *
   * <pre>
   * ip_1bit = sum_d(q_byte[d] * sign[d])
   * factorIp_1bit = -2 * sqrt(sqrX) / (x0 * sqrt(paddedDim))
   * factorPpc_1bit = factorIp_1bit * (2*popcount - paddedDim) / sqrt(paddedDim)
   * dist = sqrX + sqrY + factorPpc_1bit*vl + (2*ip_1bit - sumQ)*factorIp_1bit*widthOver255
   * </pre>
   *
   * @param query the query vector (must have length == dimension)
   * @param similarityFunction the similarity function to use
   * @return a score function using only 1-bit sign codes (Layer 1)
   */
  public ScoreFunction coarseScoreFunctionFor(
      float[] query, SimilarityFunction similarityFunction) {
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Expected query dimension " + dimension + ", got " + query.length);
    }
    PreparedQuery pq = prepareQuery(query);

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal -> {
            float dist =
                estimateL2Layer1(
                    signCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    pq.sqrtPaddedDim());
            return 1f / (1f + Math.max(dist, 0f));
          };
      case DOT_PRODUCT ->
          ordinal -> {
            float dist =
                estimateL2Layer1(
                    signCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    pq.sqrtPaddedDim());
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            return Math.max((1f + approxDot) / 2f, 0f);
          };
      case COSINE ->
          ordinal -> {
            float dist =
                estimateL2Layer1(
                    signCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    pq.sqrtPaddedDim());
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float vecNorm = (float) Math.sqrt(sqrX);
            if (pq.queryNorm() == 0f || vecNorm == 0f) return 0f;
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            float cosine = approxDot / (pq.queryNorm() * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            float dist =
                estimateL2Layer1(
                    signCodes[ordinal],
                    corrections[ordinal],
                    pq.queryBytes(),
                    pq.vl(),
                    pq.widthOver255(),
                    pq.sumQ(),
                    pq.sqrY(),
                    pq.sqrtPaddedDim());
            float sqrX = corrections[ordinal][ExtendedRaBitQuantizer.IDX_SQR_X];
            float approxDot = (pq.sqrY() + sqrX - dist) / 2f;
            return SimilarityFunction.scaleMaxInnerProductScore(approxDot);
          };
    };
  }

  /**
   * Estimates squared L2 distance using only 1-bit sign codes (Layer 1). Uses the same formula as
   * 1-bit RaBitQ, with correction factors derived from {@code sqrX} and {@code x0} (stored per
   * vector) and the sign code popcount (computed on the fly).
   *
   * <p>This is cheaper than {@link #estimateL2Multi} because it skips the magnitude unpacking step,
   * at the cost of a less accurate distance estimate.
   *
   * @param storedSignBits packed sign codes as longs
   * @param corrections per-vector corrections (uses IDX_SQR_X and IDX_X0)
   * @param queryBytes uint8-quantized rotated query [0, 255]
   * @param vl minimum value of the rotated query
   * @param widthOver255 quantization step size
   * @param sumQ sum of all quantized query values
   * @param sqrY query's squared distance to centroid
   * @param sqrtPaddedDim sqrt(paddedDimension), precomputed once per query
   * @return estimated squared L2 distance (coarse, 1-bit accuracy)
   */
  static float estimateL2Layer1(
      long[] storedSignBits,
      float[] corrections,
      byte[] queryBytes,
      float vl,
      float widthOver255,
      int sumQ,
      float sqrY,
      float sqrtPaddedDim) {
    float sqrX = corrections[ExtendedRaBitQuantizer.IDX_SQR_X];
    float x0 = corrections[ExtendedRaBitQuantizer.IDX_X0];

    // Derive 1-bit correction factors from sqrX and x0 (same as 1-bit RaBitQ)
    float sqrtSqrX = (float) Math.sqrt(sqrX);
    float factorIp1bit = (x0 > 0f) ? -2.0f * sqrtSqrX / (x0 * sqrtPaddedDim) : 0f;

    // Popcount: number of 1-bits in the sign codes
    // factorPpc = factorIp * (2*popcount - paddedDim), same formula as 1-bit RaBitQ
    int popcount = totalBitCount(storedSignBits);
    int paddedDim = queryBytes.length;
    float factorPpc1bit = factorIp1bit * (2.0f * popcount - paddedDim);

    // 1-bit asymmetric IP: sum_d(q_byte[d] * sign[d])
    int ip1bit = RaBitQuantizedVectors.asymmetricByteTimesBit(queryBytes, storedSignBits);

    return sqrX + sqrY + factorPpc1bit * vl + (2 * ip1bit - sumQ) * factorIp1bit * widthOver255;
  }

  /**
   * Estimates squared L2 distance using the multi-bit Extended RaBitQ formula (Layer 2).
   *
   * <p>Uses the same structure as 1-bit RaBitQ: {@code sqrX + sqrY + factorPpc*vl +
   * factorIp*widthOver255*B} where B is the signed multi-bit asymmetric IP: {@code
   * sum_d(queryBytes[d] * (2*sign[d]-1) * mag[d])}. The correction factors factorPpc and factorIp
   * incorporate the multi-bit code quality (x0_multi > x0_1bit), giving tighter estimates.
   *
   * <p>The +0.5 offset from the greedy quantization is incorporated via the 1-bit IP: {@code B_full
   * = B_mag + 0.5*(2*ip_1bit - sumQ)}.
   *
   * @param storedSignBits packed sign codes as longs
   * @param storedMagCodes packed magnitude codes (bits per dimension)
   * @param corrections per-vector corrections [sqrX, x0, factorPpc, factorIp, errorFactor, ...]
   * @param queryBytes uint8-quantized rotated query [0, 255]
   * @param vl minimum value of the rotated query (lower quantization bound)
   * @param widthOver255 quantization step size (max - min) / 255
   * @param sumQ sum of all quantized query values
   * @param sqrY query's squared distance to centroid
   * @param bits magnitude bit-width
   * @return estimated squared L2 distance
   */
  static float estimateL2Multi(
      long[] storedSignBits,
      byte[] storedMagCodes,
      float[] corrections,
      byte[] queryBytes,
      float vl,
      float widthOver255,
      int sumQ,
      float sqrY,
      int bits) {
    float sqrX = corrections[ExtendedRaBitQuantizer.IDX_SQR_X];
    float factorPpc = corrections[ExtendedRaBitQuantizer.IDX_FACTOR_PPC];
    float factorIp = corrections[ExtendedRaBitQuantizer.IDX_FACTOR_IP];

    // Signed multi-bit IP: sum_d(q_byte[d] * (2*sign[d]-1) * mag[d])
    int signedMagIp =
        asymmetricByteTimesSignedMag(queryBytes, storedMagCodes, storedSignBits, bits);

    // 1-bit IP for the +0.5 offset contribution: sum_d(q_byte[d] * sign[d])
    int ip1bit = RaBitQuantizedVectors.asymmetricByteTimesBit(queryBytes, storedSignBits);

    // B_full = signedMagIp + 0.5 * (2*ip_1bit - sumQ)
    // The +0.5 accounts for the offset in the greedy quantize code: code[d] = (mag+0.5)/rescale
    float bFull = signedMagIp + 0.5f * (2 * ip1bit - sumQ);

    return sqrX + sqrY + factorPpc * vl + factorIp * widthOver255 * bFull;
  }

  /**
   * Computes the signed asymmetric inner product between uint8-quantized query bytes and multi-bit
   * magnitude codes with sign information: {@code sum_d(queryBytes[d] * (2*sign[d]-1) * mag[d])}.
   *
   * <p>This is the multi-bit analog of RaBitQ's {@code (2*ip_1bit - sumQ)} term. Where 1-bit RaBitQ
   * uses only the sign ({@code ±1}), multi-bit adds magnitude precision ({@code ±mag[d]}).
   *
   * @param queryBytes uint8-quantized query [0, 255]
   * @param magCodes packed magnitude codes (B bits per dimension)
   * @param signBits packed sign codes as longs
   * @param bits magnitude bit-width
   * @return the signed asymmetric inner product
   */
  static int asymmetricByteTimesSignedMag(
      byte[] queryBytes, byte[] magCodes, long[] signBits, int bits) {
    int mask = (1 << bits) - 1;
    int result = 0;
    int bitPos = 0;
    int dim = queryBytes.length;

    for (int d = 0; d < dim; d++) {
      int byteIdx = bitPos / 8;
      int bitOffset = bitPos % 8;

      int magIdx;
      if (bitOffset + bits <= 8) {
        magIdx = ((magCodes[byteIdx] & 0xFF) >> bitOffset) & mask;
      } else {
        int lo = (magCodes[byteIdx] & 0xFF) >>> bitOffset;
        int hi = (magCodes[byteIdx + 1] & 0xFF) << (8 - bitOffset);
        magIdx = (lo | hi) & mask;
      }

      int longIdx = d >> 6;
      int signBitIdx = d & 63;
      boolean positive = ((signBits[longIdx] >> signBitIdx) & 1) == 1;

      int qVal = queryBytes[d] & 0xFF;
      if (positive) {
        result += qVal * magIdx;
      } else {
        result -= qVal * magIdx;
      }

      bitPos += bits;
    }
    return result;
  }

  /** Returns the sign codes for the vector at the given ordinal. */
  public long[] getSignCodes(int ordinal) {
    return signCodes[ordinal];
  }

  /** Returns the magnitude codes for the vector at the given ordinal. */
  public byte[] getMagCodes(int ordinal) {
    return magCodes[ordinal];
  }

  /** Returns the correction factors for the vector at the given ordinal. */
  public float[] getCorrections(int ordinal) {
    return corrections[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public ExtendedRaBitQuantizer quantizer() {
    return quantizer;
  }

  /**
   * Returns the total number of 1-bits across all longs in the given packed bit array. Used by
   * Layer 1 scoring to compute the sign-bit popcount needed for the 1-bit factorPpc correction.
   *
   * @param bits packed sign codes as longs
   * @return the total number of set bits
   */
  static int totalBitCount(long[] bits) {
    int count = 0;
    for (long l : bits) {
      count += Long.bitCount(l);
    }
    return count;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
