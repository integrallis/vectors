package com.integrallis.vectors.ivf;

import java.util.List;

/**
 * The result of an {@link IvfIndex#search} call.
 *
 * @param hits ordered list of top-k hits, descending by score; may contain fewer than k results
 *     when the index has fewer than k vectors or the minScore filter is active
 * @param clustersSearched number of clusters that were actually scanned (after SOAR expansion)
 */
public record IvfSearchResult(List<IvfHit> hits, int clustersSearched) {

  /** Returns {@code true} when no hits were found. */
  public boolean isEmpty() {
    return hits.isEmpty();
  }

  /** Number of hits returned (≤ k). */
  public int size() {
    return hits.size();
  }
}
