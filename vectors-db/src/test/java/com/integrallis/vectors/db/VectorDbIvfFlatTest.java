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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * P18 gate: VectorCollection with {@link IndexType#IVF_FLAT} — recall, persistence, metadata
 * filters, and deletions.
 */
@Tag("unit")
class VectorDbIvfFlatTest {

  private static final int DIM = 32;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private VectorCollection buildIvf(int k, int nprobe, Path storage) {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.IVF_FLAT)
            .ivfK(k)
            .ivfNprobe(nprobe);
    if (storage != null) builder.storagePath(storage);
    return builder.build();
  }

  private int[] bruteForceTopK(float[] query, float[][] data, int k) {
    record P(int i, float d) {}
    P[] ps = new P[data.length];
    for (int i = 0; i < data.length; i++) {
      float d = 0f;
      for (int j = 0; j < query.length; j++) {
        float diff = query[j] - data[i][j];
        d += diff * diff;
      }
      ps[i] = new P(i, d);
    }
    Arrays.sort(ps, (a, b) -> Float.compare(a.d(), b.d()));
    int[] r = new int[k];
    for (int i = 0; i < k; i++) r[i] = ps[i].i();
    return r;
  }

  // ─── recall ──────────────────────────────────────────────────────────────

  @Test
  void buildAndSearch_recallExceedsThreshold() {
    float[][] data = randomVecs(500, DIM, 1L);
    try (var col = buildIvf(4, 3, null)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.commit();

      int queryCount = 30, k = 10;
      double totalRecall = 0.0;
      for (int q = 0; q < queryCount; q++) {
        float[] query = randomVecs(1, DIM, 1000L + q)[0];
        List<SearchResult.Hit> hits = col.search(SearchRequest.builder(query, k).build()).hits();
        Set<Integer> gtSet = new HashSet<>();
        for (int g : bruteForceTopK(query, data, k)) gtSet.add(g);
        long found =
            hits.stream()
                .filter(h -> gtSet.contains(Integer.parseInt(h.id().substring(4))))
                .count();
        totalRecall += (double) found / k;
      }
      assertThat(totalRecall / queryCount).isGreaterThanOrEqualTo(0.75);
    }
  }

  // ─── size ────────────────────────────────────────────────────────────────

  @Test
  void sizeReflectsAdds() {
    try (var col = buildIvf(4, 2, null)) {
      assertThat(col.size()).isZero();
      for (int i = 0; i < 50; i++) col.add(Document.of("doc-" + i, randomVecs(1, DIM, i)[0]));
      col.commit();
      assertThat(col.size()).isEqualTo(50);
    }
  }

  // ─── filter ──────────────────────────────────────────────────────────────

  @Test
  void search_appliesMetadataFilter() {
    float[][] data = randomVecs(60, DIM, 2L);
    try (var col = buildIvf(4, 4, null)) {
      for (int i = 0; i < data.length; i++) {
        String color = (i % 2 == 0) ? "red" : "blue";
        col.add(new Document("doc-" + i, data[i], null, Map.of("color", MetadataValue.of(color))));
      }
      col.commit();

      float[] query = data[0];
      List<SearchResult.Hit> hits =
          col.search(SearchRequest.builder(query, 10).filter(Filters.eq("color", "red")).build())
              .hits();
      assertThat(hits).isNotEmpty();
      // All returned hits must have color=red
      hits.forEach(
          h -> {
            int idx = Integer.parseInt(h.id().substring(4));
            assertThat(idx % 2).isZero();
          });
    }
  }

  // ─── delete ──────────────────────────────────────────────────────────────

  @Test
  void search_withDelete_excludesDeletedDoc() {
    float[][] data = randomVecs(50, DIM, 3L);
    // Put the needle as a very distinctive vector
    float[] needle = new float[DIM];
    Arrays.fill(needle, 100f);
    try (var col = buildIvf(4, 4, null)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.add(Document.of("needle", needle));
      col.commit();

      // Before delete: needle is top hit
      List<SearchResult.Hit> before = col.search(SearchRequest.builder(needle, 1).build()).hits();
      assertThat(before).isNotEmpty();
      assertThat(before.get(0).id()).isEqualTo("needle");

      // After delete + commit: needle must not appear
      assertThat(col.delete("needle")).isTrue();
      col.commit(); // tombstones are applied on commit, matching FLAT/HNSW/VAMANA semantics
      List<SearchResult.Hit> after = col.search(SearchRequest.builder(needle, 5).build()).hits();
      assertThat(after.stream().map(SearchResult.Hit::id)).doesNotContain("needle");
    }
  }

  // ─── persistence ─────────────────────────────────────────────────────────

  @Test
  void commit_thenReopen_preservesVectors(@TempDir Path tmp) {
    float[][] data = randomVecs(200, DIM, 4L);
    float[] needle = new float[DIM];
    Arrays.fill(needle, 50f);

    // Session 1: build, add, commit, close
    try (var col = buildIvf(4, 3, tmp)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.add(Document.of("needle", needle));
      col.commit();
    }

    // Session 2: reopen, search
    try (var col = buildIvf(4, 3, tmp)) {
      List<SearchResult.Hit> hits = col.search(SearchRequest.builder(needle, 1).build()).hits();
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }

  // -----------------------------------------------------------------------
  // DP4: Two-pass over-query expansion
  // -----------------------------------------------------------------------

  @Test
  void twoPassSearch_overQueryFactor_improvesRecall() {
    // 300 random vectors; brute-force ground truth for k=5
    int n = 300;
    float[][] data = randomVecs(n, DIM, 42L);
    float[] query = randomVecs(1, DIM, 77L)[0];
    int k = 5;

    int[] gt = bruteForceTopK(query, data, k);
    Set<Integer> gtSet = new HashSet<>();
    for (int idx : gt) gtSet.add(idx);

    try (var col = buildIvf(10, 2, null)) {
      for (int i = 0; i < n; i++) col.add(Document.of(String.valueOf(i), data[i]));
      col.commit();

      // Single-pass (overQueryFactor = 1): baseline recall
      var single = col.search(SearchRequest.builder(query, k).overQueryFactor(1.0f).build()).hits();
      long singleRecall =
          single.stream().filter(h -> gtSet.contains(Integer.parseInt(h.id()))).count();

      // Two-pass (overQueryFactor = 4): probe 4× more clusters, retrieve 4× more candidates
      var twoPass =
          col.search(SearchRequest.builder(query, k).overQueryFactor(4.0f).build()).hits();
      long twoPassRecall =
          twoPass.stream().filter(h -> gtSet.contains(Integer.parseInt(h.id()))).count();

      // Two-pass must return exactly k results and must not degrade recall vs single-pass
      assertThat(twoPass).hasSize(k);
      assertThat(twoPassRecall).isGreaterThanOrEqualTo(singleRecall);
    }
  }

  @Test
  void twoPassSearch_explicitFactor_matchesDefault() {
    // Explicitly setting overQueryFactor(4.0f) must be identical to the implicit default of 4.0f.
    int n = 100;
    float[][] data = randomVecs(n, DIM, 10L);
    float[] query = randomVecs(1, DIM, 20L)[0];

    try (var col = buildIvf(5, 2, null)) {
      for (int i = 0; i < n; i++) col.add(Document.of(String.valueOf(i), data[i]));
      col.commit();

      var defaultReq = SearchRequest.builder(query, 5).build(); // default overQueryFactor = 4.0f
      var explicit4Req = SearchRequest.builder(query, 5).overQueryFactor(4.0f).build();

      var defaultHits = col.search(defaultReq).hits().stream().map(SearchResult.Hit::id).toList();
      var explicit4Hits =
          col.search(explicit4Req).hits().stream().map(SearchResult.Hit::id).toList();

      assertThat(explicit4Hits).containsExactlyElementsOf(defaultHits);
    }
  }

  @Test
  void twoPassSearch_highFactor_returnsExactlyK() {
    // overQueryFactor=10 should still return exactly k hits, not more.
    int n = 200;
    float[][] data = randomVecs(n, DIM, 30L);
    float[] query = randomVecs(1, DIM, 50L)[0];

    try (var col = buildIvf(8, 2, null)) {
      for (int i = 0; i < n; i++) col.add(Document.of(String.valueOf(i), data[i]));
      col.commit();

      var result = col.search(SearchRequest.builder(query, 5).overQueryFactor(10.0f).build());
      assertThat(result.hits()).hasSize(5);
    }
  }
}
