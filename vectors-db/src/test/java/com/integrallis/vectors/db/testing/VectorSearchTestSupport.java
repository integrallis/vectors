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
package com.integrallis.vectors.db.testing;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Shared test utilities for vector search tests.
 *
 * <p>Provides deterministic document generation, brute-force recall computation, and randomized
 * configuration selection. Inspired by Elasticsearch's {@code InternalTestCluster} and Apache
 * Ignite's {@code ConfigVariationsTestSuiteBuilder}.
 *
 * <p>All methods are static and stateless. Randomized methods accept an explicit {@link Random} for
 * reproducibility — callers should seed with a known value and log the seed on failure.
 */
public final class VectorSearchTestSupport {

  /** System property for overriding the random seed ({@code -Dvectors.test.seed=12345}). */
  public static final String SEED_PROPERTY = "vectors.test.seed";

  private VectorSearchTestSupport() {}

  // ---------------------------------------------------------------------------
  // Random seed management
  // ---------------------------------------------------------------------------

  /**
   * Returns a test seed: either from the {@code vectors.test.seed} system property or from {@link
   * System#nanoTime()}. The caller should log this seed so that failures are reproducible.
   */
  public static long testSeed() {
    String prop = System.getProperty(SEED_PROPERTY);
    if (prop != null && !prop.isBlank()) {
      return Long.parseLong(prop.trim());
    }
    return System.nanoTime();
  }

  // ---------------------------------------------------------------------------
  // Document generation
  // ---------------------------------------------------------------------------

  /** Generates a random float vector with values in {@code [-0.5, 0.5)}. */
  public static float[] randomVector(int dimension, Random rng) {
    float[] v = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      v[i] = rng.nextFloat() - 0.5f;
    }
    return v;
  }

  /** Generates a unit-normalized random float vector. */
  public static float[] randomUnitVector(int dimension, Random rng) {
    float[] v = randomVector(dimension, rng);
    float norm = 0f;
    for (float x : v) {
      norm += x * x;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < v.length; i++) {
        v[i] /= norm;
      }
    }
    return v;
  }

  /** Generates {@code count} documents with random vectors and optional metadata. */
  public static List<Document> generateDocs(int count, int dimension, long seed) {
    Random rng = new Random(seed);
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      docs.add(new Document("doc-" + i, randomVector(dimension, rng), "text-" + i, null));
    }
    return docs;
  }

  /** Generates documents with metadata fields useful for filter testing. */
  public static List<Document> generateDocsWithMetadata(int count, int dimension, long seed) {
    Random rng = new Random(seed);
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      Map<String, MetadataValue> md = new HashMap<>();
      md.put("idx", MetadataValue.of((long) i));
      md.put("name", MetadataValue.of("doc-" + i));
      md.put("flag", MetadataValue.of(i % 2 == 0));
      md.put("group", MetadataValue.of("group-" + (i % 5)));
      docs.add(new Document("doc-" + i, randomVector(dimension, rng), "text-" + i, md));
    }
    return docs;
  }

  // ---------------------------------------------------------------------------
  // Recall computation
  // ---------------------------------------------------------------------------

  /**
   * Computes brute-force top-k document ids by the given similarity function.
   *
   * @return set of top-k document ids
   */
  public static Set<String> bruteForceTopKIds(
      List<Document> docs, float[] query, int k, SimilarityFunction metric) {
    int n = docs.size();
    String[] ids = new String[n];
    float[] scores = new float[n];
    for (int i = 0; i < n; i++) {
      ids[i] = docs.get(i).id();
      scores[i] = metric.compare(query, docs.get(i).vector());
    }
    Set<String> top = new HashSet<>();
    boolean[] used = new boolean[n];
    for (int r = 0; r < Math.min(k, n); r++) {
      int best = -1;
      for (int i = 0; i < n; i++) {
        if (used[i]) continue;
        if (best == -1 || scores[i] > scores[best]) best = i;
      }
      if (best == -1) break;
      used[best] = true;
      top.add(ids[best]);
    }
    return top;
  }

  /** Computes recall@k as the fraction of ground-truth ids found in the search result. */
  public static double measureRecall(
      List<Document> docs, float[] query, int k, SimilarityFunction metric, SearchResult result) {
    Set<String> groundTruth = bruteForceTopKIds(docs, query, k, metric);
    if (groundTruth.isEmpty()) return 1.0;
    long overlap = 0;
    for (SearchResult.Hit hit : result.hits()) {
      if (groundTruth.contains(hit.id())) overlap++;
    }
    return (double) overlap / groundTruth.size();
  }

  /**
   * Convenience: search the collection and return recall against brute-force ground truth.
   *
   * @param docs all documents in the collection
   * @param query search query vector
   * @param k top-k
   * @param metric similarity function
   * @param collection the vector collection to search
   * @return recall in [0.0, 1.0]
   */
  public static double searchAndMeasureRecall(
      List<Document> docs,
      float[] query,
      int k,
      SimilarityFunction metric,
      VectorCollection collection) {
    SearchResult result = collection.search(SearchRequest.builder(query, k).build());
    return measureRecall(docs, query, k, metric, result);
  }

  // ---------------------------------------------------------------------------
  // Randomized configuration
  // ---------------------------------------------------------------------------

  /** Picks a random element from the given array. */
  public static <T> T randomFrom(T[] values, Random rng) {
    return values[rng.nextInt(values.length)];
  }

  /** Picks a random similarity function. */
  public static SimilarityFunction randomMetric(Random rng) {
    return randomFrom(SimilarityFunction.values(), rng);
  }

  /**
   * Picks a random index type from the graph-based set (HNSW, VAMANA). Excludes FLAT (trivial) and
   * IVF variants (require specific configuration).
   */
  public static IndexType randomGraphIndexType(Random rng) {
    return rng.nextBoolean() ? IndexType.HNSW : IndexType.VAMANA;
  }

  /** Picks a random quantizer kind (including NONE). */
  public static QuantizerKind randomQuantizer(Random rng) {
    return randomFrom(QuantizerKind.values(), rng);
  }

  /**
   * Picks a random quantizer kind that is compatible with graph indexes (HNSW/VAMANA). All
   * quantizers are compatible with graph indexes.
   */
  public static QuantizerKind randomGraphQuantizer(Random rng) {
    return randomFrom(QuantizerKind.values(), rng);
  }

  /** Picks a random HNSW M value from typical choices. */
  public static int randomHnswM(Random rng) {
    int[] choices = {8, 16, 32, 64};
    return choices[rng.nextInt(choices.length)];
  }

  /** Picks a random HNSW efConstruction value from typical choices. */
  public static int randomHnswEfConstruction(Random rng) {
    int[] choices = {100, 200, 400};
    return choices[rng.nextInt(choices.length)];
  }

  /** Picks a random dimension from typical choices. */
  public static int randomDimension(Random rng) {
    int[] choices = {32, 64, 128, 256};
    return choices[rng.nextInt(choices.length)];
  }

  /**
   * Creates a randomly-configured collection builder. The caller must set {@code dimension} and
   * {@code metric} explicitly (or use the overload that accepts a seed).
   */
  public static VectorCollectionBuilder randomBuilder(Random rng) {
    IndexType indexType = randomGraphIndexType(rng);
    QuantizerKind quantizer = randomQuantizer(rng);

    VectorCollectionBuilder builder =
        VectorCollection.builder().indexType(indexType).quantizer(quantizer);

    if (indexType == IndexType.HNSW) {
      builder.hnswM(randomHnswM(rng)).hnswEfConstruction(randomHnswEfConstruction(rng));
    }

    return builder;
  }

  /**
   * Creates a fully-configured random collection builder with dimension and metric included.
   *
   * @param rng seeded Random for reproducibility
   * @return builder ready for {@code .build()} or further customization
   */
  public static VectorCollectionBuilder randomBuilderComplete(Random rng) {
    return randomBuilder(rng).dimension(randomDimension(rng)).metric(randomMetric(rng));
  }
}
