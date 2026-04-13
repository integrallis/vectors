package com.integrallis.vectors.db;

import java.util.List;
import java.util.Objects;

/**
 * The result of a {@link VectorCollection#search(SearchRequest)} call: an ordered list of hits plus
 * the wall-clock search time in nanoseconds.
 *
 * <p>{@code searchTimeNanos} measures the <b>end-to-end</b> search latency observed by the caller:
 * SPI traversal, post-filter execution, metadata hydration, and result projection. It does
 * <b>not</b> include acquisition of the collection's read lock (usually negligible).
 *
 * @param hits ranked hits, descending by score
 * @param searchTimeNanos end-to-end elapsed time for the search in nanoseconds
 */
public record SearchResult(List<Hit> hits, long searchTimeNanos) {

  public SearchResult {
    Objects.requireNonNull(hits, "hits must not be null");
    hits = List.copyOf(hits);
  }

  /**
   * A single ranked result.
   *
   * @param id the document id
   * @param score similarity score
   * @param document the backing document (with vector / text / metadata honouring the request's
   *     include flags)
   */
  public record Hit(String id, float score, Document document) {
    public Hit {
      Objects.requireNonNull(id, "id must not be null");
      Objects.requireNonNull(document, "document must not be null");
    }
  }
}
