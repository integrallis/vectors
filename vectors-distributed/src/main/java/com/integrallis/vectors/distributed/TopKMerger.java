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
 * <p>De-duplicates by document id (keeping the highest score), sorts descending by score, and
 * truncates to k. Thread-safe (stateless, pure function).
 */
public final class TopKMerger {

  private TopKMerger() {}

  /**
   * Merges partial results from multiple nodes.
   *
   * @param partialResults results collected from each responding node
   * @param k maximum number of hits to return
   * @param totalSearchTimeNanos combined search time to record in the merged result
   * @return merged {@link SearchResult} with at most k hits, sorted descending by score
   */
  public static SearchResult merge(
      List<SearchResult> partialResults, int k, long totalSearchTimeNanos) {
    if (partialResults.isEmpty()) {
      return new SearchResult(List.of(), totalSearchTimeNanos);
    }

    // De-duplicate by id, keeping max score
    Map<String, Hit> best = new HashMap<>();
    for (SearchResult result : partialResults) {
      for (Hit hit : result.hits()) {
        best.merge(hit.id(), hit, (a, b) -> a.score() >= b.score() ? a : b);
      }
    }

    // Use a min-heap of size k to find top-k in O(n log k)
    PriorityQueue<Hit> heap = new PriorityQueue<>(k + 1, Comparator.comparingDouble(Hit::score));
    for (Hit hit : best.values()) {
      heap.offer(hit);
      if (heap.size() > k) {
        heap.poll(); // evict lowest score
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
}
