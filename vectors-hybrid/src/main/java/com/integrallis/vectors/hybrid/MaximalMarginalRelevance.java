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
package com.integrallis.vectors.hybrid;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;

/**
 * Maximal Marginal Relevance (Carbonell &amp; Goldstein, 1998) diversity re-ranking for RAG
 * retrieval. Given a candidate set, greedily selects results that are relevant to the query while
 * penalizing redundancy with already-selected results — trading a little relevance for diversity so
 * near-duplicate chunks don't crowd the LLM context window.
 *
 * <p>At each step it picks the candidate maximizing
 *
 * <pre>{@code lambda * sim(query, c) - (1 - lambda) * max_{s in selected} sim(c, s)}</pre>
 *
 * with both terms on the same similarity scale (computed via the supplied {@link
 * SimilarityFunction}). {@code lambda = 1} is pure relevance (identical to sorting by score);
 * {@code lambda = 0} maximizes diversity. Typical RAG values are 0.5–0.7.
 *
 * <p>Cost is {@code O(k * n * dim)} for {@code n} candidates — cheap, because MMR runs over a small
 * over-fetched candidate set, not the whole corpus.
 */
public final class MaximalMarginalRelevance {

  private MaximalMarginalRelevance() {}

  /**
   * Selects up to {@code k} candidate indices in MMR order.
   *
   * @param query the query vector
   * @param candidates candidate vectors (e.g. the over-fetched top results)
   * @param k number of results to select (clamped to the candidate count)
   * @param lambda relevance/diversity trade-off in {@code [0, 1]} (1 = pure relevance)
   * @param similarity the metric for both query-relevance and inter-candidate redundancy
   * @return selected candidate indices, best-first; length {@code min(k, candidates.length)}
   */
  public static int[] select(
      float[] query, float[][] candidates, int k, float lambda, SimilarityFunction similarity) {
    validateArguments(query, candidates, lambda, similarity);
    int n = candidates.length;
    int target = Math.min(Math.max(k, 0), n);
    int[] selected = new int[target];
    if (target == 0) {
      return selected;
    }

    // COSINE has no fused kernel. Normalize once, then use the fused DOT_PRODUCT path because dot
    // equals cosine for unit vectors and has the same score transform.
    SimilarityFunction metric = similarity;
    float[] scoreQuery = query;
    float[][] pool = candidates;
    if (similarity == SimilarityFunction.COSINE) {
      metric = SimilarityFunction.DOT_PRODUCT;
      scoreQuery = VectorUtil.l2normalize(query.clone(), false);
      pool = normalizedCopy(candidates);
    }

    // Fused-SIMD batch scoring: one kernel sweep over the whole candidate pool instead of n scalar
    // compares — for query-relevance and, each step, for the newly selected candidate's redundancy.
    // FusedSimilarity applies the exact same transform as SimilarityFunction#compare.
    float[] scratch = new float[n];
    float[] relToQuery = new float[n];
    FusedSimilarity.bulkCompare(metric, scoreQuery, pool, scratch, relToQuery, n);

    boolean[] taken = new boolean[n];
    // Highest similarity of each candidate to any already-selected candidate (the redundancy term).
    float[] maxSimToSelected = new float[n];
    Arrays.fill(maxSimToSelected, Float.NEGATIVE_INFINITY);
    float[] simToBest = new float[n];

    for (int step = 0; step < target; step++) {
      int best = bestCandidate(taken, relToQuery, maxSimToSelected, step, lambda);
      selected[step] = best;
      taken[best] = true;
      // Fold the newly selected candidate into every remaining candidate's redundancy — one fused
      // sweep of sim(selected, pool).
      FusedSimilarity.bulkCompare(metric, pool[best], pool, scratch, simToBest, n);
      updateRedundancy(taken, simToBest, maxSimToSelected);
    }
    return selected;
  }

  private static void validateArguments(
      float[] query, float[][] candidates, float lambda, SimilarityFunction similarity) {
    if (query == null || candidates == null || similarity == null) {
      throw new IllegalArgumentException("query, candidates, and similarity must be non-null");
    }
    if (lambda < 0f || lambda > 1f) {
      throw new IllegalArgumentException("lambda must be in [0, 1], got " + lambda);
    }
  }

  private static float[][] normalizedCopy(float[][] candidates) {
    float[][] normalized = new float[candidates.length][];
    for (int i = 0; i < candidates.length; i++) {
      normalized[i] = VectorUtil.l2normalize(candidates[i].clone(), false);
    }
    return normalized;
  }

  private static int bestCandidate(
      boolean[] taken, float[] relevance, float[] maxSimilarity, int step, float lambda) {
    int best = -1;
    float bestScore = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < taken.length; i++) {
      if (taken[i]) {
        continue;
      }
      float redundancy = step == 0 ? 0f : maxSimilarity[i];
      float score = lambda * relevance[i] - (1f - lambda) * redundancy;
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  private static void updateRedundancy(boolean[] taken, float[] similarity, float[] maxSimilarity) {
    for (int i = 0; i < taken.length; i++) {
      if (!taken[i] && similarity[i] > maxSimilarity[i]) {
        maxSimilarity[i] = similarity[i];
      }
    }
  }
}
