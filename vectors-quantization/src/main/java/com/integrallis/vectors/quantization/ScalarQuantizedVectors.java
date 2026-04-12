package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;

/**
 * Compressed vector storage produced by {@link ScalarQuantizer}. Stores quantized byte arrays and
 * per-vector correction factors for accurate approximate scoring.
 *
 * <p>Supports both int8 (one byte per dimension) and int4 (packed, two nibbles per byte) modes.
 * Scoring uses SIMD-accelerated byte dot products (via {@link VectorUtil}) combined with correction
 * terms:
 *
 * <ul>
 *   <li><b>DOT_PRODUCT / COSINE:</b> {@code score = (1 + alpha² * byteDot + queryCorrection +
 *       vectorCorrection) / 2}
 *   <li><b>EUCLIDEAN:</b> {@code score = 1 / (1 + alpha² * byteSquareDist)}
 *   <li><b>MAXIMUM_INNER_PRODUCT:</b> {@code dot = alpha² * byteDot + queryCorrection +
 *       vectorCorrection}, then piecewise scaling
 * </ul>
 *
 * <p>For int4 mode, packed stored vectors are unpacked to full byte arrays before scoring. The
 * returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 */
public final class ScalarQuantizedVectors implements CompressedVectors {

  private final ScalarQuantizer quantizer;
  private final byte[][] quantizedVectors;
  private final float[] corrections;
  private final int dimension;

  /**
   * Constructs a {@code ScalarQuantizedVectors} from pre-encoded data. Public for cross-module
   * construction by deserialization codecs.
   *
   * @param quantizer the scalar quantizer that produced these vectors
   * @param quantizedVectors per-vector quantized byte arrays
   * @param corrections per-vector correction factors
   * @param dimension the original vector dimension
   */
  public ScalarQuantizedVectors(
      ScalarQuantizer quantizer, byte[][] quantizedVectors, float[] corrections, int dimension) {
    this.quantizer = quantizer;
    this.quantizedVectors = quantizedVectors;
    this.corrections = corrections;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return quantizedVectors.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction) {
    return switch (quantizer.bits()) {
      case INT8 -> scoreFunctionInt8(query, similarityFunction);
      case INT4 -> scoreFunctionInt4(query, similarityFunction);
    };
  }

  private ScoreFunction scoreFunctionInt8(float[] query, SimilarityFunction similarityFunction) {
    byte[] quantizedQuery = new byte[dimension];
    float queryCorrection = quantizer.encode(query, quantizedQuery);
    float constMultiplier = quantizer.constMultiplier();

    return switch (similarityFunction) {
      case DOT_PRODUCT, COSINE ->
          ordinal -> {
            int rawDot = VectorUtil.dotProduct(quantizedQuery, quantizedVectors[ordinal]);
            float adjustedDot = rawDot * constMultiplier + queryCorrection + corrections[ordinal];
            return Math.max((1f + adjustedDot) / 2f, 0f);
          };
      case EUCLIDEAN ->
          ordinal -> {
            int sqDist = VectorUtil.squareDistance(quantizedQuery, quantizedVectors[ordinal]);
            float adjustedDist = sqDist * constMultiplier;
            return 1f / (1f + adjustedDist);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            int rawDot = VectorUtil.dotProduct(quantizedQuery, quantizedVectors[ordinal]);
            float adjustedDot = rawDot * constMultiplier + queryCorrection + corrections[ordinal];
            return SimilarityFunction.scaleMaxInnerProductScore(adjustedDot);
          };
    };
  }

  private ScoreFunction scoreFunctionInt4(float[] query, SimilarityFunction similarityFunction) {
    // Encode query as unpacked int4 (one byte per dim, values [0, 15])
    byte[] quantizedQuery = new byte[dimension];
    float queryCorrection = quantizer.encodeInt4Unpacked(query, quantizedQuery);
    float constMultiplier = quantizer.constMultiplier();

    // Buffer for unpacking stored vectors (not thread-safe — one ScoreFunction per thread)
    byte[] unpacked = new byte[dimension];

    return switch (similarityFunction) {
      case DOT_PRODUCT, COSINE ->
          ordinal -> {
            quantizer.unpackInt4(quantizedVectors[ordinal], unpacked);
            int rawDot = VectorUtil.dotProduct(quantizedQuery, unpacked);
            float adjustedDot = rawDot * constMultiplier + queryCorrection + corrections[ordinal];
            return Math.max((1f + adjustedDot) / 2f, 0f);
          };
      case EUCLIDEAN ->
          ordinal -> {
            quantizer.unpackInt4(quantizedVectors[ordinal], unpacked);
            int sqDist = VectorUtil.squareDistance(quantizedQuery, unpacked);
            float adjustedDist = sqDist * constMultiplier;
            return 1f / (1f + adjustedDist);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            quantizer.unpackInt4(quantizedVectors[ordinal], unpacked);
            int rawDot = VectorUtil.dotProduct(quantizedQuery, unpacked);
            float adjustedDot = rawDot * constMultiplier + queryCorrection + corrections[ordinal];
            return SimilarityFunction.scaleMaxInnerProductScore(adjustedDot);
          };
    };
  }

  /** Returns the quantized bytes for the vector at the given ordinal. */
  public byte[] getQuantizedVector(int ordinal) {
    return quantizedVectors[ordinal];
  }

  /** Returns the correction factor for the vector at the given ordinal. */
  public float getCorrection(int ordinal) {
    return corrections[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public ScalarQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
