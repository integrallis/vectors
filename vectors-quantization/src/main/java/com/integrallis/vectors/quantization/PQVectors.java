package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;

/**
 * Compressed vector storage produced by {@link ProductQuantizer}. Stores M-byte PQ codes per vector
 * and supports fast approximate scoring via ADC (Asymmetric Distance Computation).
 *
 * <p>ADC builds a query-specific lookup table of partial distances between query sub-vectors and
 * codebook centroids. Scoring a stored vector then reduces to M table lookups and additions, making
 * it independent of the original vector dimensionality.
 *
 * <p>Scoring formulas:
 *
 * <ul>
 *   <li><b>DOT_PRODUCT:</b> {@code score = (1 + sum(table[m][code[m]])) / 2}
 *   <li><b>EUCLIDEAN:</b> {@code score = 1 / (1 + sum(table[m][code[m]]))} where table stores
 *       squared L2 partial distances
 *   <li><b>COSINE:</b> dot products and centroid norms both computed in centered space for
 *       consistency when global centering is active
 *   <li><b>MAXIMUM_INNER_PRODUCT:</b> raw dot product with piecewise scaling
 * </ul>
 *
 * <p>The returned {@link ScoreFunction} is <b>not</b> thread-safe; create one per thread.
 */
public final class PQVectors implements CompressedVectors {

  private final ProductQuantizer quantizer;
  private final byte[][] codes; // codes[i] has M bytes, one cluster index per subspace
  private final int dimension;

  /**
   * Centroid squared norms precomputed once at construction time. Shape:
   * [numSubspaces][numClusters]. Codebooks are immutable after training so these never change.
   */
  private final float[][] centroidNormSq;

  PQVectors(ProductQuantizer quantizer, byte[][] codes, int dimension) {
    this.quantizer = quantizer;
    this.codes = codes;
    this.dimension = dimension;
    this.centroidNormSq = precomputeCentroidNormsSq();
  }

  /**
   * Precomputes {@code |c_k|^2} for every centroid k in every subspace m. This O(M * Ks * subDim)
   * work is performed once and reused by every call to {@link #cosineScoreFunction}.
   */
  private float[][] precomputeCentroidNormsSq() {
    int m = quantizer.numSubspaces();
    int ks = quantizer.numClusters();
    float[][] norms = new float[m][ks];
    for (int s = 0; s < m; s++) {
      int[] sizeAndOffset = quantizer.subspaceSizeAndOffset(s);
      int subDim = sizeAndOffset[0];
      float[] codebook = quantizer.codebook(s);
      for (int k = 0; k < ks; k++) {
        float norm = 0;
        int base = k * subDim;
        for (int d = 0; d < subDim; d++) {
          norm += codebook[base + d] * codebook[base + d];
        }
        norms[s][k] = norm;
      }
    }
    return norms;
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
    return switch (similarityFunction) {
      case DOT_PRODUCT -> dotProductScoreFunction(query);
      case EUCLIDEAN -> euclideanScoreFunction(query);
      case COSINE -> cosineScoreFunction(query);
      case MAXIMUM_INNER_PRODUCT -> mipScoreFunction(query);
    };
  }

  private ScoreFunction dotProductScoreFunction(float[] query) {
    float[][] table = quantizer.buildADCTable(query, true);
    int m = quantizer.numSubspaces();

    return ordinal -> {
      byte[] code = codes[ordinal];
      float sum = 0;
      for (int s = 0; s < m; s++) {
        sum += table[s][code[s] & 0xFF];
      }
      // Convert raw dot to [0, 1] score: (1 + dot) / 2
      return Math.max((1f + sum) / 2f, 0f);
    };
  }

  private ScoreFunction euclideanScoreFunction(float[] query) {
    float[][] table = quantizer.buildADCTable(query, false);
    int m = quantizer.numSubspaces();

    return ordinal -> {
      byte[] code = codes[ordinal];
      float sumSqDist = 0;
      for (int s = 0; s < m; s++) {
        sumSqDist += table[s][code[s] & 0xFF];
      }
      return 1f / (1f + sumSqDist);
    };
  }

  private ScoreFunction cosineScoreFunction(float[] query) {
    // buildADCTable subtracts globalCentroid internally; we need the same centered query for
    // the norm so that numerator and denominator are in the same (centered) space.
    float[] gc = quantizer.globalCentroid();
    float[] centeredQuery = query;
    if (gc != null) {
      centeredQuery = Arrays.copyOf(query, query.length);
      for (int i = 0; i < query.length; i++) {
        centeredQuery[i] -= gc[i];
      }
    }

    float[][] dotTable = quantizer.buildADCTable(query, true);
    int m = quantizer.numSubspaces();

    // Query norm in centered space — consistent with the dot table's numerator (query - gc) · c_k
    float queryNormSq = 0;
    for (float v : centeredQuery) queryNormSq += v * v;
    float finalQueryNorm = (float) Math.sqrt(queryNormSq);

    return ordinal -> {
      byte[] code = codes[ordinal];
      float dot = 0;
      float vecNormSq = 0;
      for (int s = 0; s < m; s++) {
        int k = code[s] & 0xFF;
        dot += dotTable[s][k];
        vecNormSq += centroidNormSq[s][k]; // precomputed at construction
      }
      float vecNorm = (float) Math.sqrt(vecNormSq);
      if (finalQueryNorm == 0 || vecNorm == 0) return 0f;
      float cosine = dot / (finalQueryNorm * vecNorm);
      // Convert cosine [-1, 1] to score [0, 1]: (1 + cos) / 2
      return Math.max((1f + cosine) / 2f, 0f);
    };
  }

  private ScoreFunction mipScoreFunction(float[] query) {
    float[][] table = quantizer.buildADCTable(query, true);
    int m = quantizer.numSubspaces();

    return ordinal -> {
      byte[] code = codes[ordinal];
      float sum = 0;
      for (int s = 0; s < m; s++) {
        sum += table[s][code[s] & 0xFF];
      }
      return SimilarityFunction.scaleMaxInnerProductScore(sum);
    };
  }

  /** Returns the PQ code (cluster indices) for the vector at the given ordinal. */
  public byte[] getCode(int ordinal) {
    return codes[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public ProductQuantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
