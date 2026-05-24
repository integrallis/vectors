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
package com.integrallis.vectors.core;

import java.util.Objects;

/**
 * Shared fused-SIMD similarity scoring for batches of vectors. Each call loads the query SIMD chunk
 * once and applies it to four rows simultaneously via {@link VectorUtil#batchSquaredL2} / {@link
 * VectorUtil#batchDotProduct}.
 *
 * <p>Callers pass a reusable {@code float[][] pool} already populated with aliased row references
 * and a {@code scratch} buffer the same length as {@code pool}. The final similarity-score
 * transform (matching {@link SimilarityFunction#compare}) is applied into {@code outScores}.
 *
 * <p>The {@code COSINE} path has no fused SIMD kernel yet; it falls back to scalar {@link
 * SimilarityFunction#compare} per row.
 */
public final class FusedSimilarity {

  private FusedSimilarity() {}

  /**
   * Scores {@code pool[0..count)} against {@code query} using fused SIMD and writes
   * similarity-function-transformed values into {@code outScores[0..count)}.
   *
   * @param sim similarity function
   * @param query query vector
   * @param pool reusable row-reference pool (rows must be aliased, not copies)
   * @param scratch raw-kernel output buffer — at least {@code count} entries
   * @param outScores final transformed scores — at least {@code count} entries
   * @param count number of active rows in {@code pool}
   */
  public static void bulkCompare(
      SimilarityFunction sim,
      float[] query,
      float[][] pool,
      float[] scratch,
      float[] outScores,
      int count) {
    Objects.requireNonNull(sim, "sim");
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(pool, "pool");
    Objects.requireNonNull(scratch, "scratch");
    Objects.requireNonNull(outScores, "outScores");
    if (count < 0 || count > pool.length) {
      throw new IllegalArgumentException(
          "count must be in [0, pool.length]: " + count + " for " + pool.length);
    }
    if (scratch.length < count) {
      throw new IllegalArgumentException(
          "scratch.length must be >= count: " + scratch.length + " < " + count);
    }
    if (outScores.length < count) {
      throw new IllegalArgumentException(
          "outScores.length must be >= count: " + outScores.length + " < " + count);
    }
    switch (sim) {
      case EUCLIDEAN -> {
        VectorUtil.batchSquaredL2(query, pool, scratch, count);
        for (int i = 0; i < count; i++) outScores[i] = 1f / (1f + scratch[i]);
      }
      case DOT_PRODUCT -> {
        VectorUtil.batchDotProduct(query, pool, scratch, count);
        for (int i = 0; i < count; i++) outScores[i] = (1f + scratch[i]) * 0.5f;
      }
      case MAXIMUM_INNER_PRODUCT -> {
        VectorUtil.batchDotProduct(query, pool, scratch, count);
        for (int i = 0; i < count; i++)
          outScores[i] = SimilarityFunction.scaleMaxInnerProductScore(scratch[i]);
      }
      case COSINE -> {
        for (int i = 0; i < count; i++) outScores[i] = sim.compare(query, pool[i]);
      }
    }
  }
}
