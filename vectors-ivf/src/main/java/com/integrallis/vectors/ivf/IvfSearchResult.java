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
