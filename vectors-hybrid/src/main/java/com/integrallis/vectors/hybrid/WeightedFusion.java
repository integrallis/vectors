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
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * Weighted score fusion with min-max normalization.
 *
 * <p>Each retriever's scores are first normalized to [0, 1] via min-max scaling, then multiplied by
 * the retriever's weight. The final fused score for a document is the sum of its weighted
 * normalized scores across all retrievers.
 */
public final class WeightedFusion implements FusionStrategy {

  private final float[] weights;

  /**
   * Creates a weighted fusion strategy.
   *
   * @param weights per-retriever weights (must match the number of result lists passed to {@link
   *     #fuse}). Weights are used as-is (not normalized).
   * @throws IllegalArgumentException if weights is empty or contains non-positive values
   */
  public WeightedFusion(float... weights) {
    Objects.requireNonNull(weights, "weights");
    if (weights.length == 0) {
      throw new IllegalArgumentException("at least one weight is required");
    }
    for (int i = 0; i < weights.length; i++) {
      if (weights[i] <= 0f) {
        throw new IllegalArgumentException("weight[" + i + "] must be positive, got " + weights[i]);
      }
    }
    this.weights = weights.clone();
  }

  @Override
  public List<ScoredId> fuse(List<List<ScoredId>> results, int k) {
    if (k <= 0) {
      return List.of();
    }
    if (results.size() != weights.length) {
      throw new IllegalArgumentException(
          "expected " + weights.length + " result lists, got " + results.size());
    }

    Map<String, Float> scores = new HashMap<>();
    for (int i = 0; i < results.size(); i++) {
      List<ScoredId> list = results.get(i);
      if (list.isEmpty()) {
        continue;
      }
      float min = Float.MAX_VALUE;
      float max = -Float.MAX_VALUE;
      for (ScoredId hit : list) {
        min = Math.min(min, hit.score());
        max = Math.max(max, hit.score());
      }
      float range = max - min;
      float weight = weights[i];
      for (ScoredId hit : list) {
        float normalized = range > 0f ? (hit.score() - min) / range : 1.0f;
        scores.merge(hit.id(), normalized * weight, Float::sum);
      }
    }

    return topK(scores, k);
  }

  /**
   * Selects the top-{@code k} entries ordered by score descending, ties broken by id ascending —
   * bit-identical to a full sort followed by {@code limit(k)}, but in {@code O(n log k)} time using
   * a bounded size-{@code k} min-heap instead of sorting the entire {@code n}-entry union.
   */
  private static List<ScoredId> topK(Map<String, Float> scores, int k) {
    // Min-heap head is the "weakest" kept entry: lowest score, ties broken by largest id. When the
    // heap exceeds k, that weakest entry is evicted, leaving the strongest k.
    Comparator<Map.Entry<String, Float>> weakestFirst =
        Comparator.<Map.Entry<String, Float>, Float>comparing(Map.Entry::getValue)
            .thenComparing(Map.Entry::getKey, Comparator.reverseOrder());
    PriorityQueue<Map.Entry<String, Float>> heap = new PriorityQueue<>(weakestFirst);
    for (Map.Entry<String, Float> e : scores.entrySet()) {
      if (heap.size() < k) {
        heap.offer(e);
      } else if (weakestFirst.compare(e, heap.peek()) > 0) {
        heap.poll();
        heap.offer(e);
      }
    }
    // Draining the heap yields weakest-first; fill the result back-to-front to get the final
    // strongest-first (score desc, id asc) ordering.
    ScoredId[] out = new ScoredId[heap.size()];
    for (int i = out.length - 1; i >= 0; i--) {
      Map.Entry<String, Float> e = heap.poll();
      out[i] = new ScoredId(e.getKey(), e.getValue());
    }
    return List.of(out);
  }
}
