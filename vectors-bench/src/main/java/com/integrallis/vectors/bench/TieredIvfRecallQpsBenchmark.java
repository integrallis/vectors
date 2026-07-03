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
package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.ivf.DistributedVectorCollection;
import com.integrallis.vectors.ivf.IvfBuildParams;
import com.integrallis.vectors.ivf.IvfHit;
import com.integrallis.vectors.ivf.TierPolicy;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Recall-vs-QPS sweep for the <b>tiered IVF</b> (P1.3): sweeps {@code nprobe} on a {@link
 * DistributedVectorCollection} and reports recall@10 vs steady-state QPS. Mirrors the methodology
 * of {@link RecallQpsBenchmark} but targets the tiered IVF substrate ({@code BuoyIndex} + {@code
 * TieredCluster} + {@code HyperDoor}) that landed under the "JVM turbopuffer" positioning.
 *
 * <p>Self-contained: generates a mixture-of-Gaussians synthetic corpus so the bench runs without
 * any external dataset download or network access. The IVF clustering quality therefore reflects
 * the cluster substrate, not dataset-loading plumbing — accept the numbers as relative comparisons
 * across {@code nprobe} settings, not as absolute claims against a real corpus.
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :vectors-bench:runTieredIvfRecallQps
 * </pre>
 */
public final class TieredIvfRecallQpsBenchmark {

  private static final int K = 10;
  private static final int WARMUP_ROUNDS = 3;
  private static final int MEASUREMENT_ROUNDS = 5;
  private static final int DEFAULT_DIM = 128;
  private static final int DEFAULT_CORPUS = 50_000;
  private static final int DEFAULT_QUERIES = 200;

  private TieredIvfRecallQpsBenchmark() {}

  public static void main(String[] args) throws IOException {
    int dim = Integer.getInteger("bench.tieredIvf.dim", DEFAULT_DIM);
    int corpusSize = Integer.getInteger("bench.tieredIvf.corpus", DEFAULT_CORPUS);
    int querySize = Integer.getInteger("bench.tieredIvf.queries", DEFAULT_QUERIES);

    System.out.printf(
        "[tieredIvfRecallQps] dim=%d corpus=%,d queries=%,d k=%d%n", dim, corpusSize, querySize, K);

    long t0 = System.nanoTime();
    Corpus corpus = generateCorpus(corpusSize, querySize, dim);
    System.out.printf("  generated corpus in %.2fs%n", (System.nanoTime() - t0) / 1_000_000_000.0);

    int[][] groundTruth = bruteForceTopK(corpus.queries, corpus.vectors, K);
    System.out.printf("  computed brute-force ground truth%n");

    Path walDir = Files.createTempDirectory("tiered-ivf-bench-");
    HeapStorageBackend t3 = new HeapStorageBackend();
    IvfBuildParams params = IvfBuildParams.defaults(corpusSize);
    // High thresholds keep clusters in the heap-resident scan path (no tier transitions). The
    // bench is measuring routing+scan, not tier-promotion behavior.
    TierPolicy tierPolicy = new TierPolicy(Integer.MAX_VALUE - 1, 1);
    long buildStart = System.nanoTime();
    DistributedVectorCollection coll;
    try {
      coll =
          DistributedVectorCollection.build(
              corpus.vectors,
              corpus.ids,
              SimilarityFunction.DOT_PRODUCT,
              params,
              com.integrallis.vectors.ivf.ClusterSplitter.forRootK(params.k()),
              tierPolicy,
              walDir,
              t3);
    } catch (IOException e) {
      throw new IOException("Tiered IVF build failed at " + walDir, e);
    }
    System.out.printf(
        "  built tiered IVF (k=%d) in %.2fs%n",
        params.k(), (System.nanoTime() - buildStart) / 1_000_000_000.0);

    int[] nprobes = parseInts(System.getProperty("bench.tieredIvf.nprobe"), 1, 2, 4, 8, 16, 32);
    System.out.printf("  nprobe sweep: %s%n", Arrays.toString(nprobes));

    System.out.println();
    System.out.printf("%-7s %-9s %-11s %-12s%n", "nprobe", "recall@10", "qps", "latency_us");
    for (int nprobe : nprobes) {
      Result r = measure(coll, corpus.queries, groundTruth, nprobe);
      System.out.printf(
          "%-7d %-9.4f %-11.0f %-12.1f%n", nprobe, r.recall, r.qps, r.medianLatencyMicros);
    }

    coll.close();
  }

  // ─── methodology ──────────────────────────────────────────────────────────

  private static Result measure(
      DistributedVectorCollection coll, float[][] queries, int[][] groundTruth, int nprobe) {
    int numQueries = queries.length;

    // Warmup — JIT promotion and page-cache warm.
    for (int round = 0; round < WARMUP_ROUNDS; round++) {
      for (float[] q : queries) {
        coll.search(q, K, nprobe);
      }
    }
    // Intentionally do NOT System.gc() before measurement: GC pauses observed in production are
    // part of the latency distribution we want to publish (matches T4.11 hygiene).

    long[] latencies = new long[numQueries * MEASUREMENT_ROUNDS];
    int[][] resultIds = new int[numQueries][K];
    int idx = 0;
    for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
      for (int qi = 0; qi < numQueries; qi++) {
        long t0 = System.nanoTime();
        List<IvfHit> hits = coll.search(queries[qi], K, nprobe);
        latencies[idx++] = System.nanoTime() - t0;
        if (round == MEASUREMENT_ROUNDS - 1) {
          for (int j = 0; j < K; j++) {
            resultIds[qi][j] = j < hits.size() ? hits.get(j).ordinal() : -1;
          }
        }
      }
    }
    Arrays.sort(latencies);

    double medianMicros = latencies[latencies.length / 2] / 1_000.0;
    double meanSecondsPerQuery = Arrays.stream(latencies).average().orElse(0) / 1_000_000_000.0;
    double qps = meanSecondsPerQuery > 0 ? 1.0 / meanSecondsPerQuery : Double.NaN;

    double recall = recallAtK(resultIds, groundTruth);
    return new Result(recall, qps, medianMicros);
  }

  private static double recallAtK(int[][] resultIds, int[][] groundTruth) {
    int total = 0;
    int hits = 0;
    for (int qi = 0; qi < resultIds.length; qi++) {
      Set<Integer> gt = new HashSet<>();
      for (int g : groundTruth[qi]) gt.add(g);
      total += gt.size();
      for (int r : resultIds[qi]) if (r >= 0 && gt.contains(r)) hits++;
    }
    return total == 0 ? 0.0 : (double) hits / total;
  }

  // ─── corpus generation ─────────────────────────────────────────────────────

  private record Corpus(float[][] vectors, String[] ids, float[][] queries) {}

  private record Result(double recall, double qps, double medianLatencyMicros) {}

  /**
   * Generates a mixture-of-Gaussians synthetic corpus: 16 cluster centers placed on a unit sphere,
   * each emitting points with isotropic Gaussian noise. Queries are drawn from the same
   * distribution so the recall ceiling is reachable without requiring a real ANN-Benchmarks
   * dataset. Vectors are L2-normalized so DOT_PRODUCT scoring behaves like cosine.
   */
  private static Corpus generateCorpus(int n, int q, int dim) {
    final int clusters = 16;
    Random rng = new Random(0xC0FFEEL);
    float[][] centroids = new float[clusters][dim];
    for (int c = 0; c < clusters; c++) {
      for (int d = 0; d < dim; d++) centroids[c][d] = (float) rng.nextGaussian();
      l2Normalize(centroids[c]);
    }
    float[][] vectors = new float[n][];
    String[] ids = new String[n];
    for (int i = 0; i < n; i++) {
      int c = rng.nextInt(clusters);
      vectors[i] = sample(centroids[c], 0.25f, rng, dim);
      l2Normalize(vectors[i]);
      ids[i] = "doc-" + i;
    }
    float[][] queries = new float[q][];
    for (int i = 0; i < q; i++) {
      int c = rng.nextInt(clusters);
      queries[i] = sample(centroids[c], 0.25f, rng, dim);
      l2Normalize(queries[i]);
    }
    return new Corpus(vectors, ids, queries);
  }

  private static float[] sample(float[] center, float sigma, Random rng, int dim) {
    float[] v = new float[dim];
    for (int d = 0; d < dim; d++) v[d] = center[d] + (float) rng.nextGaussian() * sigma;
    return v;
  }

  private static void l2Normalize(float[] v) {
    double norm = 0;
    for (float x : v) norm += x * x;
    float n = (float) Math.sqrt(norm);
    if (n == 0f) {
      v[0] = 1f;
      return;
    }
    for (int i = 0; i < v.length; i++) v[i] /= n;
  }

  // ─── helpers ──────────────────────────────────────────────────────────────

  private static int[][] bruteForceTopK(float[][] queries, float[][] corpus, int k) {
    int[][] out = new int[queries.length][k];
    for (int qi = 0; qi < queries.length; qi++) {
      float[] q = queries[qi];
      int n = corpus.length;
      Integer[] order = new Integer[n];
      for (int i = 0; i < n; i++) order[i] = i;
      float[] scores = new float[n];
      for (int i = 0; i < n; i++) scores[i] = VectorUtil.dotProduct(q, corpus[i]);
      Arrays.sort(order, Comparator.comparingDouble((Integer a) -> -scores[a]));
      for (int j = 0; j < k; j++) out[qi][j] = order[j];
    }
    return out;
  }

  private static int[] parseInts(String csv, int... defaults) {
    if (csv == null || csv.isBlank()) return defaults;
    String[] parts = csv.split(",");
    int[] out = new int[parts.length];
    for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i].trim());
    return out;
  }
}
