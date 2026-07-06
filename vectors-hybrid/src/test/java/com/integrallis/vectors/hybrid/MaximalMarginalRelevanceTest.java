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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Spec for Maximal Marginal Relevance diversity re-ranking (written before the implementation).
 *
 * <p>MMR greedily selects, at each step, the candidate maximizing {@code lambda * sim(query, c) -
 * (1 - lambda) * max_{s in selected} sim(c, s)}. lambda=1 is pure relevance; lambda=0 maximizes
 * diversity.
 */
class MaximalMarginalRelevanceTest {

  private static final SimilarityFunction COSINE = SimilarityFunction.COSINE;

  // query direction ~[1,0]. A,B are near-duplicates and highly relevant; C is diverse + moderately
  // relevant. Relevance to query: A > B > C. A~B are mutually redundant.
  private static final float[] QUERY = {1f, 0f};
  private static final float[][] CANDS = {
    {1.0f, 0.05f}, // 0 = A  (most relevant)
    {1.0f, 0.15f}, // 1 = B  (near-duplicate of A, 2nd most relevant)
    {0.6f, 0.8f}, //  2 = C  (diverse, moderately relevant)
  };

  @Test
  @Tag("unit")
  void selectingAll_returnsPermutationOfIndices() {
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 0.5f, COSINE);
    assertThat(sel).hasSize(3);
    assertThat(sel).containsExactlyInAnyOrder(0, 1, 2);
  }

  @Test
  @Tag("unit")
  void lambdaOne_isPureRelevanceOrder() {
    // lambda=1 ignores the redundancy term -> descending relevance to query: A(0), B(1), C(2).
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 1.0f, COSINE);
    assertThat(sel).containsExactly(0, 1, 2);
  }

  @Test
  @Tag("unit")
  void highLambda_keepsRedundantNeighborAheadOfDiverseDoc() {
    // relevance-weighted: after A, the near-duplicate B still outranks the diverse C.
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 0.9f, COSINE);
    assertThat(sel[0]).isEqualTo(0); // A first (most relevant)
    assertThat(indexOf(sel, 1)).isLessThan(indexOf(sel, 2)); // B before C
  }

  @Test
  @Tag("unit")
  void lowLambda_promotesDiverseDocOverRedundantNeighbor() {
    // diversity-weighted: after A, the diverse C is promoted above the redundant B.
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 0.3f, COSINE);
    assertThat(sel[0]).isEqualTo(0); // A first (relevance still drives the first pick)
    assertThat(indexOf(sel, 2)).isLessThan(indexOf(sel, 1)); // C before B
  }

  @Test
  @Tag("unit")
  void k1_returnsSingleMostRelevant() {
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 1, 0.3f, COSINE);
    assertThat(sel).containsExactly(0);
  }

  @Test
  @Tag("unit")
  void kGreaterThanCandidates_clampsToCandidateCount() {
    int[] sel = MaximalMarginalRelevance.select(QUERY, CANDS, 99, 0.5f, COSINE);
    assertThat(sel).hasSize(3);
  }

  @Test
  @Tag("unit")
  void kZeroOrEmptyCandidates_returnsEmpty() {
    assertThat(MaximalMarginalRelevance.select(QUERY, CANDS, 0, 0.5f, COSINE)).isEmpty();
    assertThat(MaximalMarginalRelevance.select(QUERY, new float[0][], 5, 0.5f, COSINE)).isEmpty();
  }

  @Test
  @Tag("unit")
  void invalidLambda_throws() {
    assertThatThrownBy(() -> MaximalMarginalRelevance.select(QUERY, CANDS, 3, 1.5f, COSINE))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> MaximalMarginalRelevance.select(QUERY, CANDS, 3, -0.1f, COSINE))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @Tag("unit")
  void euclidean_fusedPath_lowLambdaPromotesDiverseDoc() {
    // Exercises the fused-SIMD batch path (EUCLIDEAN, unlike COSINE, is fused). Same corpus shape:
    // relevance A>B>C, A~B redundant. Low lambda must promote the diverse C above the redundant B.
    int[] hi = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 0.9f, SimilarityFunction.EUCLIDEAN);
    int[] lo = MaximalMarginalRelevance.select(QUERY, CANDS, 3, 0.3f, SimilarityFunction.EUCLIDEAN);
    assertThat(hi[0]).isEqualTo(0);
    assertThat(indexOf(hi, 1)).isLessThan(indexOf(hi, 2)); // high lambda: B before C
    assertThat(lo[0]).isEqualTo(0);
    assertThat(indexOf(lo, 2))
        .isLessThan(indexOf(lo, 1)); // low lambda: diverse C before redundant B
  }

  private static int indexOf(int[] arr, int value) {
    for (int i = 0; i < arr.length; i++) if (arr[i] == value) return i;
    return -1;
  }
}
