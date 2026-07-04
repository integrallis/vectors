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
import java.util.Arrays;

/**
 * Compressed vector storage produced by {@link BinaryQuantizer}. Stores 1-bit-per-dimension binary
 * codes as {@code long[]} arrays for efficient Hamming distance computation via {@link
 * VectorUtil#hammingDistance(long[], long[])}.
 *
 * <p>Supports two scoring modes:
 *
 * <ul>
 *   <li><b>SIGN_BIT</b>: All similarity functions reduce to Hamming distance. Fast but approximate.
 *   <li><b>BBQ</b>: Asymmetric int4-query x 1-bit-stored scoring with per-vector correction factors
 *       ({@code distToC}, {@code vl}, {@code width}). Higher recall than SIGN_BIT.
 * </ul>
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 */
public final class BinaryQuantizedVectors implements CompressedVectors {

  private final BinaryQuantizer quantizer;
  private final long[][] codes; // codes[i] has numLongs longs (dim/64, ceil)
  private final float[][] corrections; // null for SIGN_BIT; [size][3] for BBQ
  private final int dimension;

  /**
   * Constructs a {@code BinaryQuantizedVectors} from pre-encoded binary codes. Public for
   * cross-module construction by deserialization codecs.
   *
   * @param quantizer the binary quantizer that produced these codes
   * @param codes per-vector binary codes as packed longs
   * @param corrections per-vector corrections (null for SIGN_BIT, [size][3] for BBQ)
   * @param dimension the original vector dimension
   */
  public BinaryQuantizedVectors(
      BinaryQuantizer quantizer, long[][] codes, float[][] corrections, int dimension) {
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
    return switch (quantizer.mode()) {
      case SIGN_BIT -> signBitScoreFunction(query, similarityFunction);
      case BBQ -> bbqScoreFunction(query, similarityFunction);
    };
  }

  // --- SIGN_BIT scoring ---

  private ScoreFunction signBitScoreFunction(float[] query, SimilarityFunction similarityFunction) {
    // Encode query to binary long[]
    byte[] queryBytes = new byte[quantizer.encodedByteSize()];
    BinaryQuantizer.packBits(query, queryBytes, null);
    long[] queryLongs = new long[quantizer.numLongs()];
    BinaryQuantizer.bytesToLongs(queryBytes, queryLongs);

    return switch (similarityFunction) {
      case DOT_PRODUCT, COSINE ->
          ordinal -> {
            int hamming = VectorUtil.hammingDistance(queryLongs, codes[ordinal]);
            // For sign vectors: cosine = 1 - 2*hamming/dim; score = (1 + cosine) / 2
            return 1.0f - (float) hamming / dimension;
          };
      case EUCLIDEAN ->
          ordinal -> {
            int hamming = VectorUtil.hammingDistance(queryLongs, codes[ordinal]);
            return 1.0f / (1.0f + 4.0f * hamming);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            int hamming = VectorUtil.hammingDistance(queryLongs, codes[ordinal]);
            float rawDot = (float) (dimension - 2 * hamming) / dimension;
            return SimilarityFunction.scaleMaxInnerProductScore(rawDot);
          };
    };
  }

  // --- BBQ scoring ---

  private ScoreFunction bbqScoreFunction(float[] query, SimilarityFunction similarityFunction) {
    float[] centroid = quantizer.centroid();
    int numLongs = quantizer.numLongs();

    // Center query
    float[] centeredQuery = Arrays.copyOf(query, query.length);
    VectorUtil.subInPlace(centeredQuery, centroid);

    // Compute query stats
    float queryNormSq = 0f;
    float queryMin = Float.POSITIVE_INFINITY;
    float queryMax = Float.NEGATIVE_INFINITY;
    for (int d = 0; d < dimension; d++) {
      float v = centeredQuery[d];
      queryNormSq += v * v;
      if (v < queryMin) queryMin = v;
      if (v > queryMax) queryMax = v;
    }
    float queryWidth = queryMax - queryMin;
    float queryNorm = (float) Math.sqrt(queryNormSq);

    // Quantize query to int4 [0, 15]
    byte[] q4 = new byte[dimension];
    float q4Scale = queryWidth > 0 ? 15.0f / queryWidth : 0f;
    int querySum = 0;
    for (int d = 0; d < dimension; d++) {
      int q = Math.round((centeredQuery[d] - queryMin) * q4Scale);
      q = Math.max(0, Math.min(15, q));
      q4[d] = (byte) q;
      querySum += q;
    }

    // Extract 4 bit-planes from int4 query
    long[][] bitPlanes = extractBitPlanes(q4, numLongs);

    // Precompute query-level terms for the asymmetric scoring formula:
    // v'[d] ~ vl + b[d] * width
    // q'[d] ~ queryMin + q4[d] * queryWidth / 15
    // q' . v' ~ queryMin * (vl * dim + width * popCnt) + (queryWidth/15) * (vl * querySum +
    //           width * rawAsymDot)
    float queryWidthOver15 = queryWidth / 15.0f;

    int finalQuerySum = querySum;
    float finalQueryMin = queryMin;
    float finalQueryNorm = queryNorm;
    float finalQueryNormSq = queryNormSq;

    return switch (similarityFunction) {
      case DOT_PRODUCT ->
          ordinal -> {
            float approxDot =
                computeBbqDot(
                    codes[ordinal],
                    bitPlanes,
                    corrections[ordinal],
                    finalQueryMin,
                    queryWidthOver15,
                    finalQuerySum,
                    dimension);
            // approxDot is in centered space: q' . v'
            // Full dot = q . v = (q' + c) . (v' + c) = q'.v' + q'.c + c.v' + c.c
            // But we approximate: q . v ~ q' . v' + q . c + c . v' (ignoring c.c overlap)
            // Simpler: use the centered-space dot directly as an estimate for ranking
            return Math.max((1f + approxDot) / 2f, 0f);
          };
      case EUCLIDEAN ->
          ordinal -> {
            float approxDot =
                computeBbqDot(
                    codes[ordinal],
                    bitPlanes,
                    corrections[ordinal],
                    finalQueryMin,
                    queryWidthOver15,
                    finalQuerySum,
                    dimension);
            float distToC = corrections[ordinal][0];
            float adjustedDist = finalQueryNormSq + distToC - 2f * approxDot;
            return 1f / (1f + Math.max(adjustedDist, 0f));
          };
      case COSINE ->
          ordinal -> {
            float approxDot =
                computeBbqDot(
                    codes[ordinal],
                    bitPlanes,
                    corrections[ordinal],
                    finalQueryMin,
                    queryWidthOver15,
                    finalQuerySum,
                    dimension);
            float distToC = corrections[ordinal][0];
            float vecNorm = (float) Math.sqrt(distToC);
            if (finalQueryNorm == 0f || vecNorm == 0f) return 0f;
            float cosine = approxDot / (finalQueryNorm * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            // approxDot = q'·v' where q' = q - centroid, v' = v - centroid.
            // True inner product is q·v = q'·v' + q'·c + c·v' + |c|².
            // The c·v' term varies per vector and is omitted here as an accepted approximation;
            // the ranking is correct when centroid magnitude is small relative to q'·v'.
            float approxDot =
                computeBbqDot(
                    codes[ordinal],
                    bitPlanes,
                    corrections[ordinal],
                    finalQueryMin,
                    queryWidthOver15,
                    finalQuerySum,
                    dimension);
            return SimilarityFunction.scaleMaxInnerProductScore(approxDot);
          };
    };
  }

  /**
   * Computes the approximate dot product between a stored binary vector and an int4-quantized query
   * using the asymmetric BBQ formula.
   *
   * <p>The stored vector {@code v'} is approximated as {@code v'[d] ~ vl + b[d] * width}, and the
   * int4 query is approximated as {@code q'[d] ~ queryMin + q4[d] * queryWidthOver15}. The dot
   * product expands to:
   *
   * <pre>
   * q' . v' ~ queryMin * (vl * dim + width * popCnt)
   *         + queryWidthOver15 * (vl * querySum + width * rawAsymDot)
   * </pre>
   */
  static float computeBbqDot(
      long[] storedBits,
      long[][] queryBitPlanes,
      float[] corrections,
      float queryMin,
      float queryWidthOver15,
      int querySum,
      int dim) {
    float vl = corrections[1];
    float width = corrections[2];

    int popCnt = totalBitCount(storedBits);
    int rawAsymDot = asymmetricDot(storedBits, queryBitPlanes);

    return queryMin * (vl * dim + width * popCnt)
        + queryWidthOver15 * (vl * querySum + width * rawAsymDot);
  }

  /**
   * Computes the asymmetric dot product between stored 1-bit codes and 4 bit-planes extracted from
   * an int4-quantized query. The result is {@code sum_d q4[d] * b[d]} where {@code b[d]} is the
   * stored bit.
   */
  static int asymmetricDot(long[] storedBits, long[][] queryBitPlanes) {
    int result = 0;
    for (int bitPos = 0; bitPos < 4; bitPos++) {
      // SIMD popcount of (stored & plane) — same per-plane AND+BIT_COUNT primitive as Hamming.
      int planeSum = BitCounts.popcountAnd(storedBits, queryBitPlanes[bitPos]);
      result += planeSum << bitPos;
    }
    return result;
  }

  /** Returns the total number of 1-bits across all longs. */
  static int totalBitCount(long[] bits) {
    return BitCounts.popcount(bits);
  }

  /**
   * Extracts 4 bit-planes from an int4 query. Bit-plane {@code i} has bit {@code d} set iff bit
   * {@code i} of {@code q4[d]} is set.
   *
   * @param q4 the int4-quantized query (one byte per dimension, values in [0, 15])
   * @param numLongs the number of longs per bit-plane (ceil(dimension / 64))
   * @return 4 bit-planes, each as a {@code long[]} of length {@code numLongs}
   */
  static long[][] extractBitPlanes(byte[] q4, int numLongs) {
    long[][] planes = new long[4][numLongs];
    for (int d = 0; d < q4.length; d++) {
      int longIndex = d >> 6;
      long mask = 1L << (d & 63);
      int val = q4[d] & 0xFF;
      for (int bit = 0; bit < 4; bit++) {
        if (((val >> bit) & 1) == 1) {
          planes[bit][longIndex] |= mask;
        }
      }
    }
    return planes;
  }

  /** Returns the binary codes for the vector at the given ordinal. */
  public long[] getCode(int ordinal) {
    return codes[ordinal];
  }

  /** Returns the correction factors for the vector at the given ordinal, or null for SIGN_BIT. */
  public float[] getCorrections(int ordinal) {
    return corrections != null ? corrections[ordinal] : null;
  }

  /** Returns the quantizer used to create these vectors. */
  public BinaryQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
