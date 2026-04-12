package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;

/**
 * Compressed vector storage produced by {@link NVQuantizer}. Stores uint8-quantized bytes with
 * per-subvector metadata for asymmetric dequantize-on-the-fly scoring via NQT logit.
 *
 * <p>Scoring dequantizes each stored byte through the inverse NQT transform and computes the
 * distance against the full-precision query. This avoids storing fully dequantized vectors while
 * achieving near-lossless reconstruction accuracy.
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 *
 * @see NVQuantizer
 * @see NQTransform
 */
public final class NVQuantizedVectors implements CompressedVectors {

  private final NVQuantizer quantizer;
  private final byte[][] quantizedBytes; // [size][dimension]
  private final float[][] subvectorMetadata; // [size][numSubvectors * 4]
  private final int dimension;

  /**
   * Constructs a {@code NVQuantizedVectors} from pre-encoded quantized data. Public for
   * cross-module construction by deserialization codecs.
   *
   * @param quantizer the NVQ quantizer that produced these vectors
   * @param quantizedBytes per-vector quantized uint8 bytes
   * @param subvectorMetadata per-vector per-subvector metadata (alpha, x0, min, max)
   * @param dimension the original vector dimension
   */
  public NVQuantizedVectors(
      NVQuantizer quantizer, byte[][] quantizedBytes, float[][] subvectorMetadata, int dimension) {
    this.quantizer = quantizer;
    this.quantizedBytes = quantizedBytes;
    this.subvectorMetadata = subvectorMetadata;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return quantizedBytes.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction) {
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Expected query dimension " + dimension + ", got " + query.length);
    }
    int numSV = quantizer.numSubvectors();
    int[] svSizes = quantizer.subvectorSizes();
    float[] globalMean = quantizer.globalMean();

    // Center query
    float[] centeredQuery = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      centeredQuery[d] = query[d] - globalMean[d];
    }

    // Precompute query squared norm for cosine: sqrt(a*b) vs sqrt(a)*sqrt(b) saves one sqrt/score.
    float finalQueryNormSq = VectorUtil.dotProduct(centeredQuery, centeredQuery);

    // Pre-allocate the NQT inverse transform params buffer: one float[5] per scoreFunctionFor()
    // call, captured by the lambda and reused for every score(ordinal) call — zero allocations
    // in the hot scoring path regardless of similarity function.
    float[] p = new float[5];

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal ->
              1f
                  / (1f
                      + computeL2Squared(
                          centeredQuery,
                          quantizedBytes[ordinal],
                          subvectorMetadata[ordinal],
                          svSizes,
                          numSV,
                          p));
      case DOT_PRODUCT ->
          ordinal ->
              Math.max(
                  (1f
                          + computeDotProduct(
                              centeredQuery,
                              quantizedBytes[ordinal],
                              subvectorMetadata[ordinal],
                              svSizes,
                              numSV,
                              p))
                      / 2f,
                  0f);
      case COSINE ->
          // Inline dot + vecNormSq in a single pass: eliminates the float[2] allocation that
          // computeDotAndNorm() previously returned on every score call.
          ordinal -> {
            float dot = 0f;
            float vecNormSq = 0f;
            int dimOffset = 0;
            float[] meta = subvectorMetadata[ordinal];
            for (int m = 0; m < numSV; m++) {
              NVQuantizer.loadInvTransformParams(meta, m * NVQuantizer.METADATA_FLOATS_PER_SV, p);
              int svDim = svSizes[m];
              for (int d = 0; d < svDim; d++) {
                int q = quantizedBytes[ordinal][dimOffset + d] & 0xFF;
                float deq = NQTransform.scaledLogit(q / 255f, p[4], p[1], p[3], p[2]);
                dot += centeredQuery[dimOffset + d] * deq;
                vecNormSq += deq * deq;
              }
              dimOffset += svDim;
            }
            // Single sqrt(a*b) instead of sqrt(a) * sqrt(b)
            float denomSq = finalQueryNormSq * vecNormSq;
            if (denomSq == 0f) return 0f;
            float cosine = dot / (float) Math.sqrt(denomSq);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal ->
              SimilarityFunction.scaleMaxInnerProductScore(
                  computeDotProduct(
                      centeredQuery,
                      quantizedBytes[ordinal],
                      subvectorMetadata[ordinal],
                      svSizes,
                      numSV,
                      p));
    };
  }

  /**
   * Computes the squared L2 distance between {@code centeredQuery} and the dequantized stored
   * vector. Reuses the pre-allocated {@code p} buffer ({@code float[5]}) for inverse NQT transform
   * parameters — one allocation per {@link #scoreFunctionFor} call, not per score call.
   */
  private static float computeL2Squared(
      float[] centeredQuery,
      byte[] storedBytes,
      float[] metadata,
      int[] svSizes,
      int numSV,
      float[] p) {
    float dist = 0f;
    int dimOffset = 0;
    for (int m = 0; m < numSV; m++) {
      NVQuantizer.loadInvTransformParams(metadata, m * NVQuantizer.METADATA_FLOATS_PER_SV, p);
      int svDim = svSizes[m];
      for (int d = 0; d < svDim; d++) {
        int q = storedBytes[dimOffset + d] & 0xFF;
        float deq = NQTransform.scaledLogit(q / 255f, p[4], p[1], p[3], p[2]);
        float diff = centeredQuery[dimOffset + d] - deq;
        dist += diff * diff;
      }
      dimOffset += svDim;
    }
    return dist;
  }

  /**
   * Computes the dot product between {@code centeredQuery} and the dequantized stored vector,
   * reusing the pre-allocated {@code p} buffer for inverse NQT transform parameters.
   */
  private static float computeDotProduct(
      float[] centeredQuery,
      byte[] storedBytes,
      float[] metadata,
      int[] svSizes,
      int numSV,
      float[] p) {
    float dot = 0f;
    int dimOffset = 0;
    for (int m = 0; m < numSV; m++) {
      NVQuantizer.loadInvTransformParams(metadata, m * NVQuantizer.METADATA_FLOATS_PER_SV, p);
      int svDim = svSizes[m];
      for (int d = 0; d < svDim; d++) {
        int q = storedBytes[dimOffset + d] & 0xFF;
        float deq = NQTransform.scaledLogit(q / 255f, p[4], p[1], p[3], p[2]);
        dot += centeredQuery[dimOffset + d] * deq;
      }
      dimOffset += svDim;
    }
    return dot;
  }

  /** Returns the quantized bytes for the vector at the given ordinal. */
  public byte[] getQuantizedBytes(int ordinal) {
    return quantizedBytes[ordinal];
  }

  /** Returns the per-subvector metadata for the vector at the given ordinal. */
  public float[] getSubvectorMetadata(int ordinal) {
    return subvectorMetadata[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public NVQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
