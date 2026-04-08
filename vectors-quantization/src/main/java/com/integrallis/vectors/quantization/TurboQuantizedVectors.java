package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;

/**
 * Compressed vector storage produced by {@link TurboQuantizer}. Stores per-coordinate Lloyd-Max
 * quantized indices as packed bytes with per-vector norm corrections for distance estimation.
 *
 * <p>Scoring reconstructs the approximate vector via dequantization (codebook lookup + inverse
 * rotation + denormalization) and computes the distance against the query. While this is more
 * expensive than RaBitQ's bitwise scoring, it supports arbitrary bit-widths (2-8 bits) and achieves
 * near-optimal MSE distortion.
 *
 * <p><b>Allocation cost:</b> Each call to {@link ScoreFunction#score(int)} invokes {@link
 * TurboQuantizer#reconstructCentered(byte[], float)}, which allocates three {@code float[]} arrays
 * ({@code float[paddedDim]} for the dequantized rotated vector, {@code float[paddedDim]} from
 * {@link Rotation#inverseRotate}, and {@code float[dim]} for the truncated result). Scanning 10,000
 * candidates therefore produces ~30,000 short-lived heap allocations per query. Integrate with an
 * HNSW graph only after profiling; for high-throughput ANN search prefer RaBitQ (zero allocation)
 * or a pre-reconstructed batch approach.
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 *
 * @see TurboQuantizer
 */
public final class TurboQuantizedVectors implements CompressedVectors {

  private final TurboQuantizer quantizer;
  private final byte[][] indices; // indices[i] has paddedDim entries packed into bytes
  private final float[] norms; // norms[i] = ||v - centroid||
  private final int dimension;

  TurboQuantizedVectors(TurboQuantizer quantizer, byte[][] indices, float[] norms, int dimension) {
    this.quantizer = quantizer;
    this.indices = indices;
    this.norms = norms;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return indices.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction) {
    // Precompute query properties
    float[] centroid = quantizer.centroid();
    float[] centeredQuery = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      centeredQuery[d] = query[d] - centroid[d];
    }
    float sqrY = VectorUtil.dotProduct(centeredQuery, centeredQuery);
    float queryNorm = (float) Math.sqrt(sqrY);

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal -> {
            float[] reconstructed = quantizer.reconstructCentered(indices[ordinal], norms[ordinal]);
            float sqDist = VectorUtil.squareDistance(centeredQuery, reconstructed);
            return 1f / (1f + sqDist);
          };
      case DOT_PRODUCT ->
          ordinal -> {
            float[] reconstructed = quantizer.reconstructCentered(indices[ordinal], norms[ordinal]);
            float dot = VectorUtil.dotProduct(centeredQuery, reconstructed);
            // Approximate: ignoring centroid cross-terms for ANN ranking
            return Math.max((1f + dot) / 2f, 0f);
          };
      case COSINE ->
          ordinal -> {
            float[] reconstructed = quantizer.reconstructCentered(indices[ordinal], norms[ordinal]);
            float vecNorm = norms[ordinal];
            if (queryNorm == 0f || vecNorm == 0f) return 0f;
            float dot = VectorUtil.dotProduct(centeredQuery, reconstructed);
            float cosine = dot / (queryNorm * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            float[] reconstructed = quantizer.reconstructCentered(indices[ordinal], norms[ordinal]);
            float dot = VectorUtil.dotProduct(centeredQuery, reconstructed);
            return SimilarityFunction.scaleMaxInnerProductScore(dot);
          };
    };
  }

  /** Returns the quantized indices for the vector at the given ordinal. */
  public byte[] getIndices(int ordinal) {
    return indices[ordinal];
  }

  /** Returns the norm correction for the vector at the given ordinal. */
  public float getNorm(int ordinal) {
    return norms[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public TurboQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
