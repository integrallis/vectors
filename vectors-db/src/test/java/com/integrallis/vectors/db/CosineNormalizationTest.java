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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Correctness + persistence + benchmark coverage for the #A COSINE unit-normalization optimization:
 * by default COSINE collections store vectors verbatim and score them with the fused cosine kernel;
 * this verbatim path is the reference these tests compare against. The opt-in {@code
 * normalizeCosineVectors(true)} L2-unit-normalizes vectors at ingest and the index scores them with
 * DOT_PRODUCT (cosine of unit vectors equals their dot product).
 */
@Tag("unit")
class CosineNormalizationTest {

  private static float[] randomVector(Random rnd, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      // Non-unit magnitudes on purpose so normalization is observable.
      v[i] = (rnd.nextFloat() * 2f - 1f) * (1f + rnd.nextFloat() * 9f);
    }
    return v;
  }

  private static double norm(float[] v) {
    double s = 0;
    for (float x : v) {
      s += (double) x * x;
    }
    return Math.sqrt(s);
  }

  private static List<Document> randomDocs(Random rnd, int count, int dim) {
    List<Document> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      docs.add(Document.of("id-" + i, randomVector(rnd, dim)));
    }
    return docs;
  }

  /** Maps result id -> score for a single query, preserving rank order. */
  private static Map<String, Float> topK(VectorCollection c, float[] query, int k) {
    SearchResult r = c.search(SearchRequest.builder(query, k).includeVector(false).build());
    Map<String, Float> out = new LinkedHashMap<>();
    for (SearchResult.Hit h : r.hits()) {
      out.put(h.id(), h.score());
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Test 1: score + recall parity vs the reference cosine path
  // ---------------------------------------------------------------------------

  @Test
  void scoreAndRecallParityAgainstCosineReference() {
    int dim = 64;
    int n = 1500;
    int k = 10;
    int nQueries = 60;
    Random rnd = new Random(1234567L);
    List<Document> docs = randomDocs(rnd, n, dim);

    VectorCollection normalized = // #A opt-in: normalized + DOT
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(1)
            .normalizeCosineVectors(true)
            .build();
    VectorCollection reference = // default: verbatim vectors + fused cosine kernel
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(1)
            .build();
    normalized.addAll(docs);
    normalized.commit();
    reference.addAll(docs);
    reference.commit();

    int totalMatched = 0;
    int totalPossible = 0;
    double maxScoreDelta = 0.0;
    for (int q = 0; q < nQueries; q++) {
      float[] query = randomVector(rnd, dim);
      Map<String, Float> a = topK(normalized, query, k);
      Map<String, Float> b = topK(reference, query, k);
      totalPossible += b.size();
      for (Map.Entry<String, Float> e : b.entrySet()) {
        Float sa = a.get(e.getKey());
        if (sa != null) {
          totalMatched++;
          maxScoreDelta = Math.max(maxScoreDelta, Math.abs(sa - e.getValue()));
        }
      }
    }
    double overlap = (double) totalMatched / totalPossible;
    System.out.printf(
        "parity: top-%d overlap=%.4f, maxScoreDelta=%.2e over %d queries%n",
        k, overlap, maxScoreDelta, nQueries);
    // Rankings identical up to fp tie-breaks; scores identical up to normalization rounding.
    assertThat(overlap).isGreaterThanOrEqualTo(0.99);
    assertThat(maxScoreDelta).isLessThan(1e-4);

    normalized.close();
    reference.close();
  }

  // ---------------------------------------------------------------------------
  // Test 2: persistence round-trip (manifest flag restores the normalize decision)
  // ---------------------------------------------------------------------------

  @Test
  void persistenceRoundTripRestoresNormalizeDecision(@TempDir Path dir) {
    int dim = 48;
    int n = 800;
    int k = 10;
    int nQueries = 30;
    Random rnd = new Random(42L);
    List<Document> docs = randomDocs(rnd, n, dim);
    float[][] queries = new float[nQueries][];
    for (int i = 0; i < nQueries; i++) {
      queries[i] = randomVector(rnd, dim);
    }

    List<Map<String, Float>> before = new ArrayList<>();
    {
      VectorCollection c =
          VectorCollection.builder()
              .dimension(dim)
              .metric(SimilarityFunction.COSINE)
              .normalizeCosineVectors(true) // #A opt-in: normalized + DOT
              .indexType(IndexType.HNSW)
              .hnswBuildThreads(1)
              .storagePath(dir)
              .build();
      c.addAll(docs);
      c.commit();
      for (float[] query : queries) {
        before.add(topK(c, query, k));
      }
      c.close();
    }

    // Reopen from the same path — the manifest flag must restore normalized+DOT scoring.
    VectorCollection reopened =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(1)
            .storagePath(dir)
            .build();
    for (int i = 0; i < nQueries; i++) {
      Map<String, Float> after = topK(reopened, queries[i], k);
      assertThat(after.keySet()).containsExactlyElementsOf(before.get(i).keySet());
      for (Map.Entry<String, Float> e : before.get(i).entrySet()) {
        // Scores match across the persist round-trip modulo the in-memory float[] vs reopened
        // segment-kernel SIMD reduction order (one ULP) — the normalize decision is what's under
        // test.
        assertThat(after.get(e.getKey())).isCloseTo(e.getValue(), within(1e-4f));
      }
    }
    reopened.close();
  }

  // ---------------------------------------------------------------------------
  // Test 3: getVector semantics
  // ---------------------------------------------------------------------------

  @Test
  void getVectorIsVerbatimByDefaultAndUnitLengthWhenNormalized() {
    int dim = 32;
    Random rnd = new Random(7L);
    float[] original = randomVector(rnd, dim);
    assertThat(norm(original)).isGreaterThan(1.5); // ensure it is genuinely non-unit

    // Default: retrieved vector equals the original input (verbatim).
    VectorCollection verbatim =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    float[] toAdd = original.clone();
    verbatim.add(Document.of("x", toAdd));
    verbatim.commit();
    // The caller's array must not have been mutated.
    assertThat(toAdd).containsExactly(original);
    assertThat(verbatim.get("x").vector()).containsExactly(original);
    verbatim.close();

    // normalizeCosineVectors(true): stored/retrieved vector is unit length.
    VectorCollection normalized =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .normalizeCosineVectors(true)
            .build();
    normalized.add(Document.of("x", original.clone()));
    normalized.commit();
    float[] retrieved = normalized.get("x").vector();
    assertThat(norm(retrieved)).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-5));
    normalized.close();
  }

  // ---------------------------------------------------------------------------
  // Test 4: non-COSINE metric untouched (no normalization, indexMetric == metric)
  // ---------------------------------------------------------------------------

  @Test
  void nonCosineMetricIsNotNormalized() {
    int dim = 32;
    int n = 300;
    int k = 8;
    Random rnd = new Random(99L);
    List<Document> docs = randomDocs(rnd, n, dim);

    VectorCollection euclidean =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(1)
            .build();
    euclidean.addAll(docs);
    euclidean.commit();
    // Vectors are stored verbatim for EUCLIDEAN (normalizeCosineVectors is ignored).
    float[] stored = euclidean.get("id-0").vector();
    assertThat(stored).containsExactly(docs.get(0).vector());
    assertThat(norm(stored)).isGreaterThan(1.5); // definitely not unit-normalized

    // Results are identical whether or not normalizeCosineVectors is set (it is a no-op here).
    VectorCollection euclideanNormalize =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.HNSW)
            .hnswBuildThreads(1)
            .normalizeCosineVectors(true)
            .build();
    euclideanNormalize.addAll(docs);
    euclideanNormalize.commit();

    for (int q = 0; q < 20; q++) {
      float[] query = randomVector(rnd, dim);
      // normalizeCosineVectors is a no-op for EUCLIDEAN, so the returned neighbours (ranking) are
      // identical. Compare ids rather than exact score maps — search scores can differ by a ULP
      // from reduction order and that is not a regression.
      assertThat(topK(euclidean, query, k).keySet())
          .isEqualTo(topK(euclideanNormalize, query, k).keySet());
    }
    euclidean.close();
    euclideanNormalize.close();
  }

  // ---------------------------------------------------------------------------
  // Test 5: QPS microbench — normalizeCosineVectors(true) vs verbatim default
  // ---------------------------------------------------------------------------

  @Test
  void qpsMicrobench() throws IOException {
    int dim = 768;
    int n = 10_000;
    int k = 10;
    int warmup = 100;
    int measured = 1000;
    Random rnd = new Random(2024L);
    List<Document> docs = randomDocs(rnd, n, dim);

    // HNSW is the real hot path: it scores only a few, cache-warm vectors per query, so distance
    // COMPUTE dominates — where dot (1 reduction) beats cosine (3 reductions + sqrt/divide). A full
    // FLAT scan is memory-bandwidth-bound (streams every vector each query), which hides the kernel
    // win.
    VectorCollection normalized =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .normalizeCosineVectors(true)
            .build();
    VectorCollection reference =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .build();
    normalized.addAll(docs);
    normalized.commit();
    reference.addAll(docs);
    reference.commit();

    float[][] queries = new float[warmup + measured][];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = randomVector(rnd, dim);
    }

    double qpsNorm = benchmark(normalized, queries, k, warmup, measured);
    double qpsRef = benchmark(reference, queries, k, warmup, measured);
    double speedup = qpsNorm / qpsRef;

    String report =
        String.format(
            "cosine #A microbench (HNSW, n=%d, dim=%d, k=%d, measured=%d):%n"
                + "  normalized+DOT (opt-in): %.1f qps%n"
                + "  verbatim+COSINE (default reference): %.1f qps%n"
                + "  speedup: %.2fx%n",
            n, dim, k, measured, qpsNorm, qpsRef, speedup);
    System.out.print(report);
    Files.writeString(Paths.get("/tmp/cosine_bench.txt"), report);

    normalized.close();
    reference.close();
    assertThat(qpsNorm).isGreaterThan(0.0);
    assertThat(qpsRef).isGreaterThan(0.0);
  }

  private static double benchmark(
      VectorCollection c, float[][] queries, int k, int warmup, int measured) {
    for (int i = 0; i < warmup; i++) {
      c.search(SearchRequest.builder(queries[i], k).includeVector(false).build());
    }
    long start = System.nanoTime();
    for (int i = 0; i < measured; i++) {
      c.search(SearchRequest.builder(queries[warmup + i], k).includeVector(false).build());
    }
    long elapsed = System.nanoTime() - start;
    return measured / (elapsed / 1e9);
  }
}
