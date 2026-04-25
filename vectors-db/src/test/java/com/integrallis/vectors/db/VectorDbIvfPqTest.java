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
import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end tests for {@link IndexType#IVF_PQ}: in-memory build/search recall, over-query
 * expansion behaviour, and mmap persistence round-trip.
 */
@Tag("unit")
class VectorDbIvfPqTest {

  private static final int DIM = 32;

  private static float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private VectorCollection buildIvfPq(int k, int nprobe, int m, int rescoreFactor, Path storage) {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.IVF_PQ)
            .ivfK(k)
            .ivfNprobe(nprobe)
            .ivfPqSubspaces(m)
            .ivfRescoreFactor(rescoreFactor);
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

  @Test
  void buildAndSearch_recallExceedsThreshold() {
    float[][] data = randomVecs(500, DIM, 1L);
    try (var col = buildIvfPq(8, 6, 8, 4, null)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.commit();

      int queryCount = 20, k = 10;
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
      // PQ approximation + rescore with 4× wide heap: expect at least 70% recall.
      assertThat(totalRecall / queryCount).isGreaterThanOrEqualTo(0.70);
    }
  }

  @Test
  void sizeReflectsAdds() {
    try (var col = buildIvfPq(4, 2, 8, 1, null)) {
      assertThat(col.size()).isZero();
      for (int i = 0; i < 50; i++) col.add(Document.of("doc-" + i, randomVecs(1, DIM, i)[0]));
      col.commit();
      assertThat(col.size()).isEqualTo(50);
    }
  }

  @Test
  void twoPassSearch_highFactor_returnsExactlyK() {
    int n = 200;
    float[][] data = randomVecs(n, DIM, 30L);
    float[] query = randomVecs(1, DIM, 50L)[0];

    try (var col = buildIvfPq(8, 2, 8, 2, null)) {
      for (int i = 0; i < n; i++) col.add(Document.of(String.valueOf(i), data[i]));
      col.commit();

      var result = col.search(SearchRequest.builder(query, 5).overQueryFactor(10.0f).build());
      assertThat(result.hits()).hasSize(5);
    }
  }

  @Test
  void commit_thenReopen_preservesVectorsAndPqState(@TempDir Path tmp) {
    float[][] data = randomVecs(200, DIM, 4L);
    float[] needle = new float[DIM];
    Arrays.fill(needle, 50f);

    try (var col = buildIvfPq(4, 3, 8, 4, tmp)) {
      for (int i = 0; i < data.length; i++) col.add(Document.of("doc-" + i, data[i]));
      col.add(Document.of("needle", needle));
      col.commit();
    }

    try (var col = buildIvfPq(4, 3, 8, 4, tmp)) {
      List<SearchResult.Hit> hits = col.search(SearchRequest.builder(needle, 1).build()).hits();
      assertThat(hits).isNotEmpty();
      // The needle vector is extremely distinctive (all 50s) so the restored PQ index must still
      // return it as the top result — if the PQ codebook or codes failed to round-trip, the ADC
      // scoring would push the needle off the top-k.
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }

  @Test
  void builderRejectsInvalidIvfPqConfig() {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.IVF_PQ);
    org.assertj.core.api.Assertions.assertThatThrownBy(() -> builder.ivfPqSubspaces(0))
        .isInstanceOf(IllegalArgumentException.class);
    org.assertj.core.api.Assertions.assertThatThrownBy(() -> builder.ivfPqClusters(1))
        .isInstanceOf(IllegalArgumentException.class);
    org.assertj.core.api.Assertions.assertThatThrownBy(() -> builder.ivfRescoreFactor(0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
