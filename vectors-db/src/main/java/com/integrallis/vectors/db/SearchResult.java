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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import java.util.Collections;
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
    // Skip List.copyOf to avoid allocating a second Object[] copy of the hits list. The facade
    // (VectorCollectionImpl) constructs the list internally and never mutates it after passing it
    // here, so the list is effectively immutable. External callers creating SearchResult directly
    // must not pass a mutable list that will be modified afterward.
    hits = Collections.unmodifiableList(hits);
  }

  /**
   * A single ranked result from a {@link SearchResult}.
   *
   * <p><b>Score semantics.</b> {@code score} is a <em>transformed</em> normalized value where
   * higher means more similar. The exact transform depends on the {@link
   * com.integrallis.vectors.core.SimilarityFunction} configured on the collection and the vector
   * encoding (float or byte):
   *
   * <ul>
   *   <li>{@code COSINE}: {@code (1 + raw_cosine) / 2}; float range {@code [0, 1]}, byte same.
   *   <li>{@code DOT_PRODUCT} (float): {@code (1 + raw_dot) / 2}; expects unit vectors, range
   *       {@code [0, 1]}.
   *   <li>{@code DOT_PRODUCT} (byte): {@code 0.5 + raw_dot / (2 * dim * 127 * 127)}; range
   *       approximately {@code [0, 1]}.
   *   <li>{@code EUCLIDEAN}: {@code 1 / (1 + squared_distance)}; range {@code (0, 1]}.
   *   <li>{@code MAXIMUM_INNER_PRODUCT}: piecewise monotone transform; range {@code (0, +∞)}.
   * </ul>
   *
   * <p>The transformed score preserves ranking (monotonic with respect to the raw metric) but does
   * <b>not</b> equal the raw cosine, dot product, or distance. Use {@code score} for ranking and
   * thresholding; if you need the raw underlying value, compute it from the stored vector via
   * {@link com.integrallis.vectors.core.SimilarityFunction} directly.
   *
   * @param id the document id
   * @param score transformed similarity score (higher = more similar); range and semantics depend
   *     on the collection's configured metric and encoding
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
