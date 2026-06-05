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

  /**
   * Wraps already-encoded TurboQuant state. Used by {@link TurboQuantizer#encodeAll} and by the
   * persistence codec when reconstructing from disk.
   *
   * @param quantizer the quantizer whose codebook/rotation/centroid produced {@code indices}
   * @param indices per-vector packed coordinate indices ({@code indices[i].length ==
   *     quantizer.encodedByteSize()})
   * @param norms per-vector magnitudes {@code ||v - centroid||}
   * @param dimension original (unpadded) vector dimension
   */
  public TurboQuantizedVectors(
      TurboQuantizer quantizer, byte[][] indices, float[] norms, int dimension) {
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
    // The database vector is approximated by FULL reconstruction: v ≈ centroid +
    // reconstructCentered.
    // Scoring the original (un-centered) query against the full reconstruction gives the exact
    // asymmetric estimate of ⟨q, v⟩ and ‖q − v‖ with NO dropped centroid cross-terms. The earlier
    // centered-only path silently dropped the per-vector ⟨centroid, v − centroid⟩ term, which
    // biases
    // inner-product ranking (see the TurboQuant review against arXiv:2504.19874).
    float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));

    return switch (similarityFunction) {
      case EUCLIDEAN ->
          ordinal -> {
            float sqDist = VectorUtil.squareDistance(query, reconstructFull(ordinal));
            return 1f / (1f + sqDist);
          };
      case DOT_PRODUCT ->
          ordinal -> {
            float dot = VectorUtil.dotProduct(query, reconstructFull(ordinal));
            return Math.max((1f + dot) / 2f, 0f);
          };
      case COSINE ->
          ordinal -> {
            float[] full = reconstructFull(ordinal);
            float vecNorm = (float) Math.sqrt(VectorUtil.dotProduct(full, full));
            if (queryNorm == 0f || vecNorm == 0f) return 0f;
            float cosine = VectorUtil.dotProduct(query, full) / (queryNorm * vecNorm);
            return Math.max((1f + cosine) / 2f, 0f);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            float dot = VectorUtil.dotProduct(query, reconstructFull(ordinal));
            return SimilarityFunction.scaleMaxInnerProductScore(dot);
          };
    };
  }

  /** Reconstructs the full approximate vector {@code centroid + reconstructCentered(...)}. */
  private float[] reconstructFull(int ordinal) {
    float[] full = quantizer.reconstructCentered(indices[ordinal], norms[ordinal]);
    float[] centroid = quantizer.centroid();
    for (int d = 0; d < dimension; d++) {
      full[d] += centroid[d];
    }
    return full;
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
