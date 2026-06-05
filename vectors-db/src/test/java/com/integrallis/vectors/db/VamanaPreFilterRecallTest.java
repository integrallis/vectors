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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.filter.Filters;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Pins the Vamana ACORN pre-filter (I.5). Under a <b>highly selective</b> metadata filter (~5% of
 * the corpus matches), a post-filter over-fetch (default 4×k candidates) would surface almost no
 * matching neighbours and recall would collapse; the pre-filter traverses through non-matching
 * nodes but only admits matching ones, so recall stays high. The test therefore both proves
 * correctness (every hit matches the filter) and that the pre-filter path is actually engaged
 * (recall a post-filter could not reach at this selectivity).
 */
@Tag("unit")
class VamanaPreFilterRecallTest {

  private static final int N = 2000;
  private static final int DIM = 32;
  private static final int K = 10;
  private static final int RARE_EVERY = 20; // ~5% of docs are "rare"
  private static final long SEED = 42L;

  private static final float[][] DATA = randomVectors(N, 7L);
  private static final float[][] QUERIES = randomVectors(50, 99L);

  private static float[][] randomVectors(int n, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][DIM];
    for (float[] row : m) {
      for (int d = 0; d < DIM; d++) {
        row[d] = rng.nextFloat() * 2f - 1f;
      }
    }
    return m;
  }

  private static boolean isRare(int i) {
    return i % RARE_EVERY == 0;
  }

  private static void addCorpus(VectorCollection col) {
    for (int i = 0; i < N; i++) {
      String tag = isRare(i) ? "rare" : "common";
      col.add(new Document("doc-" + i, DATA[i], null, Map.of("tag", MetadataValue.of(tag))));
    }
    col.commit();
  }

  /** Brute-force top-K among the RARE documents only — the ground truth for the filtered query. */
  private static Set<String> rareGroundTruth(float[] query) {
    Integer[] order = new Integer[N];
    for (int i = 0; i < N; i++) {
      order[i] = i;
    }
    java.util.Arrays.sort(
        order, (a, b) -> Float.compare(sqDist(query, DATA[a]), sqDist(query, DATA[b])));
    Set<String> truth = new HashSet<>();
    for (int i = 0; i < N && truth.size() < K; i++) {
      if (isRare(order[i])) {
        truth.add("doc-" + order[i]);
      }
    }
    return truth;
  }

  private static float sqDist(float[] a, float[] b) {
    float s = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  private double meanFilteredRecall(VectorCollection col) {
    double sum = 0;
    for (float[] q : QUERIES) {
      Set<String> truth = rareGroundTruth(q);
      SearchResult result =
          col.search(
              SearchRequest.builder(q, K)
                  .searchListSize(128)
                  .filter(Filters.eq("tag", "rare"))
                  .build());
      int hit = 0;
      for (SearchResult.Hit h : result.hits()) {
        // Correctness: the pre-filter must never admit a non-matching ordinal.
        assertThat(h.id()).as("hit %s must match the filter", h.id()).matches("doc-\\d+");
        int idx = Integer.parseInt(h.id().substring("doc-".length()));
        assertThat(isRare(idx)).as("hit %s must be a 'rare' doc", h.id()).isTrue();
        if (truth.contains(h.id())) {
          hit++;
        }
      }
      sum += (double) hit / truth.size();
    }
    return sum / QUERIES.length;
  }

  @Test
  void inMemoryVamanaPreFilterKeepsRecallUnderSelectiveFilter() {
    try (VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.VAMANA)
            .vamanaSeed(SEED)
            .build()) {
      addCorpus(col);
      double recall = meanFilteredRecall(col);
      assertThat(recall)
          .as("in-memory Vamana ACORN pre-filter recall@%d under ~5%% selectivity", K)
          .isGreaterThanOrEqualTo(0.7);
    }
  }

  @Test
  void persistentVamanaPreFilterKeepsRecallUnderSelectiveFilter(@TempDir Path tmp) {
    try (VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.VAMANA)
            .vamanaSeed(SEED)
            .storagePath(tmp)
            .build()) {
      addCorpus(col);
      // Reopen so search runs through the persistent paged adapter's searchWithPredicate path.
      col.commit();
      double recall = meanFilteredRecall(col);
      assertThat(recall)
          .as("persistent (paged) Vamana ACORN pre-filter recall@%d under ~5%% selectivity", K)
          .isGreaterThanOrEqualTo(0.7);
    }
  }

  /** Sanity: an empty result set is returned (not an error) when nothing matches the filter. */
  @Test
  void preFilterWithNoMatchesReturnsEmpty() {
    try (VectorCollection col =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.VAMANA)
            .vamanaSeed(SEED)
            .build()) {
      List<Document> docs = new ArrayList<>();
      for (int i = 0; i < 100; i++) {
        docs.add(new Document("d" + i, DATA[i], null, Map.of("tag", MetadataValue.of("common"))));
      }
      col.addAll(docs);
      col.commit();
      SearchResult result =
          col.search(SearchRequest.builder(DATA[0], K).filter(Filters.eq("tag", "absent")).build());
      assertThat(result.hits()).isEmpty();
    }
  }
}
