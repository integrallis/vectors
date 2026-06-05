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
 * Compressed vector storage produced by {@link Fp16Quantizer}: one little-endian binary16 value per
 * coordinate (two bytes per dimension).
 *
 * <p>Scoring upcasts each stored vector back to {@code float32} on the fly and runs the
 * full-precision {@link VectorUtil} kernels against the (full-precision) query — the "query upcast
 * to FloatVector" path. Because the upcast is near-lossless, the score formulas match the {@link
 * ScalarQuantizer} convention with the correction terms set to zero:
 *
 * <ul>
 *   <li><b>DOT_PRODUCT / COSINE:</b> {@code score = max((1 + dot) / 2, 0)}
 *   <li><b>EUCLIDEAN:</b> {@code score = 1 / (1 + squaredDistance)}
 *   <li><b>MAXIMUM_INNER_PRODUCT:</b> piecewise scaling of the raw dot product
 * </ul>
 *
 * <p>The returned {@link ScoreFunction} owns a per-call scratch buffer for the upcast and is
 * therefore <b>not</b> thread-safe; create one per thread.
 */
public final class Fp16QuantizedVectors implements CompressedVectors {

  private final Fp16Quantizer quantizer;
  private final byte[][] quantizedVectors;
  private final int dimension;

  /**
   * Constructs an {@code Fp16QuantizedVectors} from pre-encoded data. Public for cross-module
   * construction by deserialization codecs.
   *
   * @param quantizer the quantizer that produced these vectors
   * @param quantizedVectors per-vector packed binary16 bytes (each {@code dimension * 2} long)
   * @param dimension the original vector dimension
   */
  public Fp16QuantizedVectors(Fp16Quantizer quantizer, byte[][] quantizedVectors, int dimension) {
    this.quantizer = quantizer;
    this.quantizedVectors = quantizedVectors;
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
    // The query stays full precision; only the stored vectors are upcast from fp16. Reused scratch
    // makes the ScoreFunction single-threaded (one per thread), matching the SPI contract.
    float[] decoded = new float[dimension];
    return switch (similarityFunction) {
      case DOT_PRODUCT, COSINE ->
          ordinal -> {
            Fp16Quantizer.decodeInto(quantizedVectors[ordinal], decoded, dimension);
            float dot = VectorUtil.dotProduct(query, decoded);
            return Math.max((1f + dot) / 2f, 0f);
          };
      case EUCLIDEAN ->
          ordinal -> {
            Fp16Quantizer.decodeInto(quantizedVectors[ordinal], decoded, dimension);
            float sqDist = VectorUtil.squareDistance(query, decoded);
            return 1f / (1f + sqDist);
          };
      case MAXIMUM_INNER_PRODUCT ->
          ordinal -> {
            Fp16Quantizer.decodeInto(quantizedVectors[ordinal], decoded, dimension);
            float dot = VectorUtil.dotProduct(query, decoded);
            return SimilarityFunction.scaleMaxInnerProductScore(dot);
          };
    };
  }

  /** Returns the packed binary16 bytes for the vector at the given ordinal. */
  public byte[] getQuantizedVector(int ordinal) {
    return quantizedVectors[ordinal];
  }

  /** Returns the quantizer used to create these vectors. */
  public Fp16Quantizer quantizer() {
    return quantizer;
  }

  @Override
  public void close() {
    // In-memory storage; nothing to close.
  }
}
