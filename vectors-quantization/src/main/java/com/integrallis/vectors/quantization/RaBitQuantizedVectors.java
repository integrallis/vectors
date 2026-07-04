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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;

/**
 * Compressed vector storage produced by {@link RaBitQuantizer}. Stores 1-bit-per-dimension binary
 * codes as {@code long[]} arrays with per-vector correction factors for unbiased RaBitQ distance
 * estimation.
 *
 * <p>Scoring uses asymmetric 8-bit query quantization: the query is centered, rotated (without
 * normalization — the query magnitude is absorbed into the quantized vl/width), then
 * scalar-quantized to uint8 [0, 255]. The approximate distance is computed using the byte-times-bit
 * inner product between the quantized query and stored binary codes, combined with per-vector
 * correction factors.
 *
 * <p><b>Distance estimation formula (Euclidean):</b>
 *
 * <pre>
 * estimatedDist = sqrX + sqrY + factorPpc * vl + (2*ip - sumQ) * factorIp * width
 * </pre>
 *
 * where:
 *
 * <ul>
 *   <li>{@code sqrX}: stored vector's squared distance to centroid
 *   <li>{@code sqrY}: query's squared distance to centroid
 *   <li>{@code ip}: asymmetric inner product sum(q_byte[d] * bit[d])
 *   <li>{@code sumQ, vl, width}: query-level quantization constants
 *   <li>{@code factorPpc, factorIp}: per-vector correction factors
 * </ul>
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 *
 * @see RaBitQuantizer
 */
public final class RaBitQuantizedVectors implements CompressedVectors {

  private final RaBitQuantizer quantizer;
  private final long[][] codes; // codes[i] has numLongs longs
  private final float[][] corrections; // [size][5]: sqrX, x0, factorPpc, factorIp, errorFactor
  private final int dimension;

  /**
   * Constructs a {@code RaBitQuantizedVectors} from pre-encoded binary codes. Public for
   * cross-module construction by deserialization codecs.
   *
   * @param quantizer the RaBitQ quantizer that produced these codes
   * @param codes per-vector binary codes as packed longs
   * @param corrections per-vector corrections [sqrX, x0, factorPpc, factorIp, errorFactor]
   * @param dimension the original vector dimension
   */
  public RaBitQuantizedVectors(
      RaBitQuantizer quantizer, long[][] codes, float[][] corrections, int dimension) {
    this.quantizer = quantizer;
    this.codes = codes;
    this.corrections = corrections;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return codes.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction) {
    int paddedDim = quantizer.paddedDimension();

    // Center query and pad to paddedDimension (no normalization — the factorIp correction
    // factors assume the query magnitude is absorbed into the quantized values vl/width)
    float[] centeredQuery = new float[paddedDim];
    System.arraycopy(query, 0, centeredQuery, 0, query.length);
    VectorUtil.subInPlace(centeredQuery, quantizer.paddedCentroid());

    // Compute sqrY = ||q - centroid||² (SIMD)
    float sqrY = VectorUtil.dotProduct(centeredQuery, centeredQuery);
    float queryNorm = (float) Math.sqrt(sqrY);

    // Rotate the unnormalized centered query (SIMD dot products per row)
    float[] queryRotated = quantizer.rotation().rotate(centeredQuery);

    // Scalar-quantize queryRotated to uint8 [0, 255]
    float vl = Float.POSITIVE_INFINITY;
    float vh = Float.NEGATIVE_INFINITY;
    for (float v : queryRotated) {
      if (v < vl) vl = v;
      if (v > vh) vh = v;
    }
    float width = vh - vl;
    float scale = width > 0 ? 255.0f / width : 0f;

    // Quantize to uint8 [0, 255] stored as byte[] (signed Java bytes, masked with & 0xFF on reads).
    // byte[] halves the per-query cache footprint vs short[]: 1 byte/dim instead of 2.
    byte[] queryBytes = new byte[paddedDim];
    int sumQ = 0;
    for (int d = 0; d < paddedDim; d++) {
      int q = Math.round((queryRotated[d] - vl) * scale);
      q = Math.max(0, Math.min(255, q));
      queryBytes[d] = (byte) q;
      sumQ += q;
    }

    // Decompose the uint8 query into 8 bit-planes once per query so the per-vector inner product
    // reduces to a sum of SIMD AND+popcounts (bit-plane decomposition, as in
    // BinaryQuantizedVectors.asymmetricDot). Bit-identical to the per-set-bit scalar gather.
    int numLongs = (paddedDim + 63) >> 6;
    long[][] queryBitPlanes = extractByteBitPlanes(queryBytes, numLongs);

    // Precompute query-level constants
    float finalVl = vl;
    float widthOver255 = width / 255.0f;
    int finalSumQ = sumQ;
    float finalSqrY = sqrY;
    float finalQueryNorm = queryNorm;

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal -> {
            float estimatedDist =
                estimateL2Squared(
                    codes[ordinal],
                    corrections[ordinal],
                    queryBitPlanes,
                    finalVl,
                    widthOver255,
                    finalSumQ,
                    finalSqrY);
            return 1f / (1f + Math.max(estimatedDist, 0f));
          };
      case DOT_PRODUCT ->
          ordinal -> {
            float estimatedDist =
                estimateL2Squared(
                    codes[ordinal],
                    corrections[ordinal],
                    queryBitPlanes,
                    finalVl,
                    widthOver255,
                    finalSumQ,
                    finalSqrY);
            float sqrX = corrections[ordinal][RaBitQuantizer.IDX_SQR_X];
            float approxDot = (finalSqrY + sqrX - estimatedDist) / 2f;
            return Math.max((1f + approxDot) / 2f, 0f);
          };
      case COSINE ->
          ordinal -> {
            float estimatedDist =
                estimateL2Squared(
                    codes[ordinal],
                    corrections[ordinal],
                    queryBitPlanes,
                    finalVl,
                    widthOver255,
                    finalSumQ,
                    finalSqrY);
            float sqrX = corrections[ordinal][RaBitQuantizer.IDX_SQR_X];
            float vecNorm = (float) Math.sqrt(sqrX);
            if (finalQueryNorm == 0f || vecNorm == 0f) return 0f;
            float approxDot = (finalSqrY + sqrX - estimatedDist) / 2f;
            float cosine = approxDot / (finalQueryNorm * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            float estimatedDist =
                estimateL2Squared(
                    codes[ordinal],
                    corrections[ordinal],
                    queryBitPlanes,
                    finalVl,
                    widthOver255,
                    finalSumQ,
                    finalSqrY);
            float sqrX = corrections[ordinal][RaBitQuantizer.IDX_SQR_X];
            // approxDot = q'·v' via the polarization identity (||q'-v'||² = ||q'||² + ||v'||² -
            // 2q'·v').
            // The true inner product q·v = q'·v' + q'·c + c·v' + |c|²; the c·v' term varies per
            // vector and is omitted — an accepted approximation for ANN ranking.
            float approxDot = (finalSqrY + sqrX - estimatedDist) / 2f;
            return SimilarityFunction.scaleMaxInnerProductScore(approxDot);
          };
    };
  }

  /**
   * Estimates the squared L2 distance between a query and a stored vector using the RaBitQ formula.
   *
   * <pre>
   * estimatedDist = sqrX + sqrY + factorPpc * vl + (2*ip - sumQ) * factorIp * widthOver255
   * </pre>
   *
   * @param storedBits the stored binary codes
   * @param corrections per-vector corrections [sqrX, x0, factorPpc, factorIp, errorFactor]
   * @param queryBitPlanes 8 bit-planes of the uint8-quantized rotated query [0, 255]
   * @param vl minimum value of the rotated query (lower quantization bound)
   * @param widthOver255 quantization step size (max - min) / 255
   * @param sumQ sum of all quantized query values
   * @param sqrY query's squared distance to centroid
   * @return estimated squared L2 distance
   */
  static float estimateL2Squared(
      long[] storedBits,
      float[] corrections,
      long[][] queryBitPlanes,
      float vl,
      float widthOver255,
      int sumQ,
      float sqrY) {
    float sqrX = corrections[RaBitQuantizer.IDX_SQR_X];
    float factorPpc = corrections[RaBitQuantizer.IDX_FACTOR_PPC];
    float factorIp = corrections[RaBitQuantizer.IDX_FACTOR_IP];

    int ip = asymmetricByteDot(storedBits, queryBitPlanes);

    return sqrX + sqrY + factorPpc * vl + (2 * ip - sumQ) * factorIp * widthOver255;
  }

  /**
   * Extracts 8 bit-planes from a uint8-quantized query. Bit-plane {@code b} has bit {@code d} set
   * iff bit {@code b} of {@code queryBytes[d]} is set. Mirrors {@code
   * BinaryQuantizedVectors.extractBitPlanes} but over 8 bits (uint8) instead of 4 (int4).
   *
   * @param queryBytes the uint8-quantized query [0, 255], one byte per dimension
   * @param numLongs the number of longs per bit-plane (ceil(paddedDimension / 64))
   * @return 8 bit-planes, each as a {@code long[]} of length {@code numLongs}
   */
  static long[][] extractByteBitPlanes(byte[] queryBytes, int numLongs) {
    long[][] planes = new long[8][numLongs];
    for (int d = 0; d < queryBytes.length; d++) {
      int val = queryBytes[d] & 0xFF;
      if (val == 0) {
        continue;
      }
      int longIndex = d >> 6;
      long mask = 1L << (d & 63);
      for (int bit = 0; bit < 8; bit++) {
        if (((val >> bit) & 1) == 1) {
          planes[bit][longIndex] |= mask;
        }
      }
    }
    return planes;
  }

  /**
   * Computes the asymmetric inner product {@code sum_d(queryBytes[d] * bit[d])} via bit-plane
   * decomposition: for each of the 8 query bit-planes, SIMD-popcount {@code (storedBits & plane)}
   * and weight it by {@code 2^bit}. Produces the same integer result as {@link
   * #asymmetricByteTimesBit} but amortizes the plane extraction across all vectors of a query and
   * replaces the per-set-bit scalar gather with vectorized AND+BIT_COUNT.
   *
   * @param storedBits the stored binary codes as packed longs
   * @param queryBitPlanes 8 bit-planes from {@link #extractByteBitPlanes}
   * @return the asymmetric inner product
   */
  static int asymmetricByteDot(long[] storedBits, long[][] queryBitPlanes) {
    int result = 0;
    for (int bit = 0; bit < 8; bit++) {
      int planeSum = BitCounts.popcountAnd(storedBits, queryBitPlanes[bit]);
      result += planeSum << bit;
    }
    return result;
  }

  /**
   * Computes the asymmetric inner product between uint8-quantized query values and stored binary
   * codes: {@code sum_d(queryBytes[d] * bit[d])} where bit[d] is 0 or 1.
   *
   * <p>Iterates over stored longs, extracting each set bit and accumulating the corresponding query
   * byte value. Retained for the Extended RaBitQ scoring paths (which mix sign and magnitude codes)
   * and as the scalar reference for {@link #asymmetricByteDot}.
   *
   * @param queryBytes the uint8-quantized query [0, 255], stored as byte[] (masked with {@code &
   *     0xFF} on read to treat as unsigned)
   * @param storedBits the stored binary codes as packed longs
   * @return the asymmetric inner product
   */
  static int asymmetricByteTimesBit(byte[] queryBytes, long[] storedBits) {
    int result = 0;
    for (int longIdx = 0; longIdx < storedBits.length; longIdx++) {
      long bits = storedBits[longIdx];
      int baseD = longIdx << 6;
      // Process set bits using bit-extraction loop (Brian Kernighan's trick)
      while (bits != 0) {
        int bitPos = Long.numberOfTrailingZeros(bits);
        result += queryBytes[baseD + bitPos] & 0xFF; // unsigned read
        bits &= bits - 1; // clear lowest set bit
      }
    }
    return result;
  }

  /** Returns the binary codes for the vector at the given ordinal. */
  public long[] getCode(int ordinal) {
    return codes[ordinal];
  }

  /** Returns the correction factors for the vector at the given ordinal. */
  public float[] getCorrections(int ordinal) {
    return corrections[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public RaBitQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
