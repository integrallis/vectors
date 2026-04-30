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

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reciprocal Rank Fusion (RRF) — a parameter-free fusion strategy.
 *
 * <p>For each document {@code d} appearing in any result list, the fused score is:
 *
 * <pre>
 *   score(d) = sum over retrievers i of 1 / (k + rank_i(d))
 * </pre>
 *
 * <p>where {@code k} is a smoothing constant (default 60) and {@code rank_i(d)} is the 1-based rank
 * of {@code d} in retriever {@code i}'s result list.
 */
public final class RRFFusion implements FusionStrategy {

  /** Default smoothing constant as recommended by the original RRF paper (Cormack et al., 2009). */
  public static final int DEFAULT_K = 60;

  private final int smoothingK;

  /** Creates an RRF fusion with the default smoothing constant ({@value DEFAULT_K}). */
  public RRFFusion() {
    this(DEFAULT_K);
  }

  /**
   * Creates an RRF fusion with a custom smoothing constant.
   *
   * @param smoothingK positive smoothing constant (higher values reduce the influence of high
   *     ranks)
   * @throws IllegalArgumentException if {@code smoothingK} is not positive
   */
  public RRFFusion(int smoothingK) {
    if (smoothingK <= 0) {
      throw new IllegalArgumentException("smoothingK must be positive, got " + smoothingK);
    }
    this.smoothingK = smoothingK;
  }

  @Override
  public List<ScoredId> fuse(List<List<ScoredId>> results, int k) {
    if (k <= 0) {
      return List.of();
    }
    Map<String, Float> scores = new HashMap<>();
    for (List<ScoredId> list : results) {
      int rank = 1;
      for (ScoredId hit : list) {
        float rrfScore = 1.0f / (smoothingK + rank);
        scores.merge(hit.id(), rrfScore, Float::sum);
        rank++;
      }
    }
    return scores.entrySet().stream()
        .sorted(
            Comparator.<Map.Entry<String, Float>, Float>comparing(Map.Entry::getValue)
                .reversed()
                .thenComparing(Map.Entry::getKey))
        .limit(k)
        .map(e -> new ScoredId(e.getKey(), e.getValue()))
        .toList();
  }
}
