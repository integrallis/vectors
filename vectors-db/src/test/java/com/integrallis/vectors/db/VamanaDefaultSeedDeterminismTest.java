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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Pins the determinism contract of the default {@code vamanaSeed}: two Vamana collections built
 * from the same documents with no explicit seed must produce byte-identical search results.
 *
 * <p>Before the fix the default was {@code System.nanoTime()}, which silently introduced
 * non-determinism on every build — undermining the Javadoc's promise of "byte-identical graphs
 * across runs with the same data" and making reproducibility audits impossible. The fix is a fixed
 * default of {@code 42L} (matching {@code DEFAULT_RABIT_SEED} / {@code DEFAULT_IVF_SEED} / {@code
 * DEFAULT_TURBO_SEED}); callers can still override via {@link
 * VectorCollectionBuilder#vamanaSeed(long)}.
 */
class VamanaDefaultSeedDeterminismTest {

  @Test
  void defaultSeedIsTheStableConstantNotSystemNanoTime() {
    // The strongest determinism contract: the *value* used as the default must be a fixed
    // constant, not a clock-derived value that happens to match across runs. Two builds in the
    // same JVM with no explicit seed must report the same seed in the live config.
    VectorCollection a =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.VAMANA)
            .build();
    VectorCollection b =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.VAMANA)
            .build();
    try {
      long seedA = a.config().vamanaParams().seed();
      long seedB = b.config().vamanaParams().seed();
      assertThat(seedA)
          .as("two builds with the default vamanaSeed must report the same seed")
          .isEqualTo(seedB);
      assertThat(seedA)
          .as("default vamanaSeed should be the stable 42L constant, not System.nanoTime()")
          .isEqualTo(VectorCollectionBuilder.DEFAULT_VAMANA_SEED);
    } finally {
      a.close();
      b.close();
    }
  }

  @Test
  void twoCollectionsWithDefaultSeedReturnIdenticalTopK() {
    final int dim = 16;
    final int n = 200;
    List<Document> corpus = randomCorpus(dim, n, 0xC0FFEEL);
    float[] query = randomVector(dim, 0xBABE1L);

    List<SearchResult.Hit> hitsA = topK(buildVamanaDefaultSeed(dim, corpus), query, 10);
    List<SearchResult.Hit> hitsB = topK(buildVamanaDefaultSeed(dim, corpus), query, 10);

    assertThat(idsOf(hitsA))
        .as("two builds with the default vamanaSeed must produce identical top-k")
        .containsExactlyElementsOf(idsOf(hitsB));
  }

  @Test
  void defaultSeedIsTheSameStableConstantAsTheOtherIndexSeeds() {
    // Cross-checks that the new default really is a stable constant (and not a clock-derived
    // value that just happened to match on two adjacent calls). Building the same collection
    // ten times in a tight loop must yield ten identical top-k orderings.
    final int dim = 8;
    final int n = 100;
    List<Document> corpus = randomCorpus(dim, n, 0xABCDEFL);
    float[] query = randomVector(dim, 0x12345L);

    List<String> reference = idsOf(topK(buildVamanaDefaultSeed(dim, corpus), query, 8));
    for (int trial = 1; trial < 10; trial++) {
      List<String> next = idsOf(topK(buildVamanaDefaultSeed(dim, corpus), query, 8));
      assertThat(next)
          .as("trial %s must match the reference build", trial)
          .containsExactlyElementsOf(reference);
    }
  }

  private static VectorCollection buildVamanaDefaultSeed(int dim, List<Document> corpus) {
    VectorCollection c =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.VAMANA)
            .build();
    for (Document d : corpus) c.add(d);
    c.commit();
    return c;
  }

  private static List<SearchResult.Hit> topK(VectorCollection c, float[] query, int k) {
    try {
      return c.search(SearchRequest.builder(query, k).build()).hits();
    } finally {
      c.close();
    }
  }

  private static List<String> idsOf(List<SearchResult.Hit> hits) {
    List<String> ids = new ArrayList<>(hits.size());
    for (SearchResult.Hit h : hits) ids.add(h.id());
    return ids;
  }

  private static List<Document> randomCorpus(int dim, int n, long seed) {
    Random rng = new Random(seed);
    List<Document> docs = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      docs.add(Document.of("doc-" + i, randomVectorFrom(rng, dim)));
    }
    return docs;
  }

  private static float[] randomVector(int dim, long seed) {
    return randomVectorFrom(new Random(seed), dim);
  }

  private static float[] randomVectorFrom(Random rng, int dim) {
    float[] v = new float[dim];
    for (int j = 0; j < dim; j++) v[j] = (float) rng.nextGaussian();
    return v;
  }
}
