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
package com.integrallis.vectors.studio.distributed.search;

import com.integrallis.vectors.studio.core.search.SearchHit;
import com.integrallis.vectors.studio.sidecart.TextSearchHit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reciprocal Rank Fusion (Cormack et al. 2009) for merging dense vector results with sparse
 * lexical/FTS results. Each candidate's fused score is {@code Σ_i 1 / (k_rrf + rank_i)} across the
 * ranked lists in which it appears; the ties-by-rank shape of RRF makes it robust to the very
 * different score scales of cosine similarity and BM25 without any normalisation.
 *
 * <p>The fuser carries forward the {@code vector} and {@code text} fields from whichever input
 * already had them, preferring the dense list (which is constructed with the sidecart's text
 * already attached for vector hits).
 */
public final class RrfFuser {

  /** Standard RRF constant, used by Elasticsearch / Vespa / OpenSearch and the original paper. */
  public static final int DEFAULT_K_RRF = 60;

  private RrfFuser() {}

  /**
   * Fuses a dense {@link SearchHit} list and a sparse {@link TextSearchHit} list into a single
   * top-{@code k} ranking via reciprocal rank fusion. Hits unique to either input are preserved.
   *
   * @param dense dense hits (already carrying vector/text where requested)
   * @param sparse sparse hits (id + bm25-style score; score itself is not used by RRF)
   * @param k maximum hits to return
   * @param kRrf RRF constant; pass {@link #DEFAULT_K_RRF} unless tuning
   * @param hydrator function from id to a fully-formed {@link SearchHit} (vector + text), used for
   *     ids that appeared only on the sparse side and therefore have no dense companion to copy
   *     fields from
   */
  public static List<SearchHit> fuse(
      List<SearchHit> dense,
      List<TextSearchHit> sparse,
      int k,
      int kRrf,
      java.util.function.Function<String, SearchHit> hydrator) {
    if (k <= 0) return List.of();
    Map<String, Double> agg = new HashMap<>();
    Map<String, SearchHit> denseById = new LinkedHashMap<>();
    for (int i = 0; i < dense.size(); i++) {
      SearchHit h = dense.get(i);
      agg.merge(h.id(), 1.0 / (kRrf + i + 1), Double::sum);
      denseById.putIfAbsent(h.id(), h);
    }
    for (int i = 0; i < sparse.size(); i++) {
      String id = sparse.get(i).id();
      agg.merge(id, 1.0 / (kRrf + i + 1), Double::sum);
    }
    List<Map.Entry<String, Double>> ordered = new ArrayList<>(agg.entrySet());
    ordered.sort(
        Comparator.<Map.Entry<String, Double>, Double>comparing(Map.Entry::getValue).reversed());
    int limit = Math.min(k, ordered.size());
    List<SearchHit> out = new ArrayList<>(limit);
    for (int i = 0; i < limit; i++) {
      Map.Entry<String, Double> e = ordered.get(i);
      String id = e.getKey();
      double rrfScore = e.getValue();
      SearchHit src = denseById.get(id);
      if (src == null && hydrator != null) src = hydrator.apply(id);
      if (src == null) {
        out.add(new SearchHit(id, rrfScore, null, null, null));
      } else {
        out.add(new SearchHit(id, rrfScore, src.vector(), src.text(), src.metadata()));
      }
    }
    return out;
  }
}
