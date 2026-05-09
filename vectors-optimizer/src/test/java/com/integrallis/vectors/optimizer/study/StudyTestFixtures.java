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
package com.integrallis.vectors.optimizer.study;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Queries;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Shared synthetic dataset for IndexStudy tests: a small, deterministic, clustered corpus where
 * each query embedding is the centroid of its labelled cluster, so any reasonable HNSW config
 * should retrieve all peers in the top-k.
 */
final class StudyTestFixtures {

  static final int DIM = 8;
  static final int N_CLUSTERS = 4;
  static final int DOCS_PER_CLUSTER = 50; // 200 docs total
  static final int QUERIES_PER_CLUSTER = 5; // 20 queries total

  private StudyTestFixtures() {}

  /** Cluster centroids: each centroid has a +1 in its own dimension and 0 elsewhere. */
  static float[] centroid(int cluster) {
    float[] v = new float[DIM];
    v[cluster % DIM] = 1.0f;
    return v;
  }

  /**
   * Builds a deterministic clustered corpus. Docs 0..{@link #RELEVANT_PER_QUERY}-1 in each cluster
   * are the closest to the centroid (tiny jitter); the remaining docs in the cluster have larger
   * jitter so they appear later in any centroid-anchored ranking.
   */
  static List<Document> corpus(long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(N_CLUSTERS * DOCS_PER_CLUSTER);
    for (int c = 0; c < N_CLUSTERS; c++) {
      float[] center = centroid(c);
      for (int i = 0; i < DOCS_PER_CLUSTER; i++) {
        float sigma = i < RELEVANT_PER_QUERY ? 0.005f : 0.15f;
        float[] v = center.clone();
        for (int d = 0; d < DIM; d++) v[d] += (float) (rng.nextGaussian() * sigma);
        String id = "doc-" + c + "-" + i;
        out.add(Document.of(id, v));
      }
    }
    return out;
  }

  /** Queries: text is "cluster-{c}-{q}"; embedding = centroid(c). */
  static Queries queries() {
    Map<String, String> q = new LinkedHashMap<>();
    for (int c = 0; c < N_CLUSTERS; c++) {
      for (int i = 0; i < QUERIES_PER_CLUSTER; i++) {
        q.put("q-" + c + "-" + i, "cluster-" + c + "-" + i);
      }
    }
    return new Queries(q);
  }

  /** How many cluster peers per query are marked relevant; recall@10 caps at 1.0 when this == 10. */
  static final int RELEVANT_PER_QUERY = 10;

  /** Qrels: each query is relevant to {@link #RELEVANT_PER_QUERY} docs in its cluster (relevance=1). */
  static Qrels qrels() {
    Map<String, Map<String, Integer>> rel = new LinkedHashMap<>();
    for (int c = 0; c < N_CLUSTERS; c++) {
      Map<String, Integer> row = new LinkedHashMap<>();
      for (int i = 0; i < RELEVANT_PER_QUERY; i++) row.put("doc-" + c + "-" + i, 1);
      for (int i = 0; i < QUERIES_PER_CLUSTER; i++) {
        rel.put("q-" + c + "-" + i, row);
      }
    }
    return new Qrels(rel);
  }

  /** Embedding provider: parses "cluster-{c}-{q}" and returns {@code centroid(c)}. */
  static EmbeddingProvider embedder() {
    return text -> {
      String[] parts = text.split("-");
      int c = Integer.parseInt(parts[1]);
      return centroid(c);
    };
  }
}
