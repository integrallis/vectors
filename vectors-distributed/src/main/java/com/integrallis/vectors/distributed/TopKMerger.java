/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.distributed;

import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.SearchResult.Hit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Merges multiple {@link SearchResult} instances from scatter-gather execution into a single top-k
 * result.
 *
 * <p>By default de-duplicates by document id (keeping the highest score), sorts descending by
 * score, and truncates to k. A {@code dedup=false} fast path skips the de-duplication pass for
 * callers whose partial-result id spaces are disjoint. Thread-safe (stateless, pure function).
 */
public final class TopKMerger {

  private TopKMerger() {}

  /**
   * Merges partial results from multiple nodes, de-duplicating by document id.
   *
   * <p>Equivalent to {@link #merge(List, int, long, boolean) merge(partialResults, k,
   * totalSearchTimeNanos, true)}. Use this on the broadcast path, where the same document id may be
   * returned by more than one responding node (e.g. replicas).
   *
   * @param partialResults results collected from each responding node
   * @param k maximum number of hits to return
   * @param totalSearchTimeNanos combined search time to record in the merged result
   * @return merged {@link SearchResult} with at most k hits, sorted descending by score
   */
  public static SearchResult merge(
      List<SearchResult> partialResults, int k, long totalSearchTimeNanos) {
    return merge(partialResults, k, totalSearchTimeNanos, true);
  }

  /**
   * Merges partial results from multiple nodes into a single top-k.
   *
   * <p>When {@code dedup} is {@code true}, duplicate document ids across partials are collapsed to
   * their highest-scoring hit before selecting top-k (an extra {@code HashMap} pass). When {@code
   * dedup} is {@code false}, that pass is skipped and hits are fed straight into the bounded
   * min-heap — a correct, allocation-lighter fast path <b>only</b> when the callers' id spaces are
   * disjoint (e.g. the cluster read path, where each document lives on exactly one shard, so no
   * shard can return an id another shard also returns). Passing {@code false} when ids can repeat
   * would let the same id appear more than once in the output.
   *
   * @param partialResults results collected from each responding node
   * @param k maximum number of hits to return
   * @param totalSearchTimeNanos combined search time to record in the merged result
   * @param dedup whether to collapse duplicate ids to their max score before top-k selection
   * @return merged {@link SearchResult} with at most k hits, sorted descending by score
   */
  public static SearchResult merge(
      List<SearchResult> partialResults, int k, long totalSearchTimeNanos, boolean dedup) {
    if (partialResults.isEmpty() || k <= 0) {
      return new SearchResult(List.of(), totalSearchTimeNanos);
    }

    // Min-heap keyed on ascending score: the root is always the weakest kept hit, so once the heap
    // holds `limit` elements we evict the root before each further offer -> O(n log k), memory
    // O(k).
    // `limit` is bounded by the number of candidate hits so a huge k never over-allocates the heap.
    int limit;
    PriorityQueue<Hit> heap;
    if (dedup) {
      // De-duplicate by id, keeping max score, then select top-k from the distinct hits.
      Map<String, Hit> best = new HashMap<>();
      for (SearchResult result : partialResults) {
        for (Hit hit : result.hits()) {
          best.merge(hit.id(), hit, (a, b) -> a.score() >= b.score() ? a : b);
        }
      }
      limit = Math.min(k, best.size());
      heap = newHeap(limit);
      for (Hit hit : best.values()) {
        offerBounded(heap, hit, limit);
      }
    } else {
      // Disjoint id space: skip the HashMap and stream hits straight into the bounded heap.
      int total = 0;
      for (SearchResult result : partialResults) {
        total += result.hits().size();
      }
      limit = Math.min(k, total);
      heap = newHeap(limit);
      for (SearchResult result : partialResults) {
        for (Hit hit : result.hits()) {
          offerBounded(heap, hit, limit);
        }
      }
    }

    // Drain heap into a list sorted descending by score
    List<Hit> hits = new ArrayList<>(heap.size());
    while (!heap.isEmpty()) {
      hits.add(heap.poll());
    }
    hits.sort(Comparator.comparingDouble(Hit::score).reversed());

    return new SearchResult(hits, totalSearchTimeNanos);
  }

  private static PriorityQueue<Hit> newHeap(int limit) {
    return new PriorityQueue<>(Math.max(1, limit + 1), Comparator.comparingDouble(Hit::score));
  }

  private static void offerBounded(PriorityQueue<Hit> heap, Hit hit, int limit) {
    heap.offer(hit);
    if (heap.size() > limit) {
      heap.poll(); // evict lowest score
    }
  }
}
