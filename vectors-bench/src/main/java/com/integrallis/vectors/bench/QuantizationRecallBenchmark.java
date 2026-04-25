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

import com.integrallis.vectors.bench.dataset.DatasetDownloader;
import com.integrallis.vectors.bench.dataset.DatasetRegistry;
import com.integrallis.vectors.bench.dataset.FvecsLoader;
import com.integrallis.vectors.bench.dataset.Hdf5Loader;
import com.integrallis.vectors.bench.report.BenchmarkReporter;
import com.integrallis.vectors.bench.report.BenchmarkResult;
import com.integrallis.vectors.bench.report.LatencyCollector;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Macro benchmark comparing quantization methods on recall, compression ratio, and search speed.
 *
 * <p>Builds HNSW indexes with different quantizers and measures two-pass search performance. This
 * benchmark produces the "compression vs quality" curves that demonstrate the value of each
 * quantization method.
 *
 * <p>Protocol:
 *
 * <ol>
 *   <li>Build HNSW index (M=32, efConstruction=200) — shared across quantizer configs
 *   <li>Enable quantization with each QuantizerKind
 *   <li>Two-pass search: coarse quantized pass + full-precision rescore
 *   <li>Measure: recall@10, QPS, latency, compression ratio
 * </ol>
 *
 * <p>Run:
 *
 * <pre>{@code
 * java --add-modules jdk.incubator.vector -cp ... QuantizationRecallBenchmark [dataset]
 * }</pre>
 */
public final class QuantizationRecallBenchmark {

  private static final int K = 10;
  private static final int M = 32;
  private static final int EF_CONSTRUCTION = 200;
  private static final int EF_SEARCH = 128;
  private static final float OVER_QUERY_FACTOR = 2.0f;
  private static final int WARMUP_ROUNDS = 3;
  private static final int MEASUREMENT_ROUNDS = 5;

  private QuantizationRecallBenchmark() {}

  /** Runs the quantization benchmark. Args: [dataset]. */
  public static void main(String[] args) throws IOException {
    String dsName = args.length > 0 ? args[0] : "sift-128-euclidean";
    SimilarityFunction sim =
        dsName.contains("angular") || dsName.contains("cosine")
            ? SimilarityFunction.COSINE
            : SimilarityFunction.EUCLIDEAN;

    System.out.printf("=== Quantization Recall Benchmark: %s ===%n", dsName);

    // Load dataset.
    float[][] corpus;
    float[][] queries;
    int[][] groundTruth;

    if (dsName.equals("sift-128-euclidean") && DatasetRegistry.isSift1MAvailable()) {
      Path dir = DatasetRegistry.sift1MDir();
      corpus = FvecsLoader.readFvecs(dir.resolve("sift_base.fvecs"));
      queries = FvecsLoader.readFvecs(dir.resolve("sift_query.fvecs"));
      groundTruth = FvecsLoader.readIvecs(dir.resolve("sift_groundtruth.ivecs"));
    } else {
      Path hdf5 = DatasetRegistry.annBenchDataset(dsName);
      if (!java.nio.file.Files.exists(hdf5) && DatasetDownloader.isKnownDataset(dsName)) {
        hdf5 = DatasetDownloader.ensureAvailable(dsName);
      }
      corpus = Hdf5Loader.readTrainVectors(hdf5);
      queries = Hdf5Loader.readTestVectors(hdf5);
      groundTruth = Hdf5Loader.readNeighbors(hdf5);
    }

    int dim = corpus[0].length;
    System.out.printf(
        "  Loaded: %,d base vectors, %,d queries, dim=%d%n", corpus.length, queries.length, dim);

    List<BenchmarkResult> results = new ArrayList<>();

    // 1) Full precision baseline (no quantization).
    results.add(benchmarkQuantizer(dsName, sim, corpus, queries, groundTruth, dim, null));

    // 2) All supported quantizer kinds.
    for (QuantizerKind qk : QuantizerKind.values()) {
      if (qk == QuantizerKind.NONE) continue;
      try {
        results.add(benchmarkQuantizer(dsName, sim, corpus, queries, groundTruth, dim, qk));
      } catch (Exception e) {
        System.out.printf("  %s: FAILED — %s%n", qk, e.getMessage());
      }
    }

    // Output.
    BenchmarkReporter.console(results, System.out);

    Path outputDir = DatasetRegistry.dataDir().resolve("results");
    BenchmarkReporter.csv(results, outputDir.resolve("quantization-recall.csv"));
    BenchmarkReporter.json(results, outputDir.resolve("quantization-recall.json"));
    System.out.printf("%nResults written to %s%n", outputDir.toAbsolutePath());
  }

  private static BenchmarkResult benchmarkQuantizer(
      String dsName,
      SimilarityFunction sim,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      QuantizerKind qk) {

    String algoName = qk == null ? "hnsw_fp32" : "hnsw_" + qk.name().toLowerCase(Locale.ROOT);
    Map<String, String> buildParams = new LinkedHashMap<>();
    buildParams.put("M", String.valueOf(M));
    buildParams.put("efConstruction", String.valueOf(EF_CONSTRUCTION));
    if (qk != null) {
      buildParams.put("quantizer", qk.name());
    }

    System.out.printf("  %s: building...", algoName);

    long t0 = System.nanoTime();
    var builder =
        VectorCollection.builder()
            .dimension(dim)
            .metric(sim)
            .indexType(IndexType.HNSW)
            .hnswM(M)
            .hnswEfConstruction(EF_CONSTRUCTION);

    if (qk != null) {
      builder.quantizer(qk);
    }

    VectorCollection col = builder.build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs, searching...", buildTime);

    try {
      // Warmup.
      for (int round = 0; round < WARMUP_ROUNDS; round++) {
        for (float[] q : queries) {
          SearchRequest.Builder req =
              SearchRequest.builder(q, K)
                  .searchListSize(EF_SEARCH)
                  .includeVector(false)
                  .includeText(false)
                  .includeMetadata(false);
          if (qk != null) {
            req.overQueryFactor(OVER_QUERY_FACTOR);
          }
          col.search(req.build());
        }
      }

      System.gc();

      // Measurement.
      LatencyCollector latency = new LatencyCollector(queries.length * MEASUREMENT_ROUNDS);
      int[][] approxResults = new int[queries.length][K];

      for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
        for (int qi = 0; qi < queries.length; qi++) {
          SearchRequest.Builder req =
              SearchRequest.builder(queries[qi], K)
                  .searchListSize(EF_SEARCH)
                  .includeVector(false)
                  .includeText(false)
                  .includeMetadata(false);
          if (qk != null) {
            req.overQueryFactor(OVER_QUERY_FACTOR);
          }

          long start = System.nanoTime();
          SearchResult result = col.search(req.build());
          latency.record(System.nanoTime() - start);

          if (round == MEASUREMENT_ROUNDS - 1) {
            List<SearchResult.Hit> hits = result.hits();
            for (int i = 0; i < Math.min(K, hits.size()); i++) {
              approxResults[qi][i] = Integer.parseInt(hits.get(i).id().substring(4));
            }
          }
        }
      }

      latency.compute();
      double recall = RecallUtil.meanRecallAtK(groundTruth, approxResults, K);

      // Estimate compression ratio.
      double compressionRatio = estimateCompression(qk, dim);

      System.out.printf(
          " recall@%d=%.3f  QPS=%,.0f  compression=%.1fx%n",
          K, recall, latency.qps(), compressionRatio);

      return BenchmarkResult.builder(dsName, algoName)
          .buildParams(Map.copyOf(buildParams))
          .searchParams(
              Map.of(
                  "efSearch", String.valueOf(EF_SEARCH),
                  "overQueryFactor", String.valueOf(OVER_QUERY_FACTOR)))
          .recall10(recall)
          .qps(latency.qps())
          .p50Us(latency.p50Us())
          .p95Us(latency.p95Us())
          .p99Us(latency.p99Us())
          .buildTimeSeconds(buildTime)
          .compressionRatio(compressionRatio)
          .build();
    } finally {
      col.close();
    }
  }

  private static double estimateCompression(QuantizerKind qk, int dim) {
    if (qk == null) return 1.0;
    double bitsPerDim =
        switch (qk) {
          case NONE -> 32;
          case SQ8 -> 8;
          case SQ4 -> 4;
          case PQ -> 8; // assumes 256 clusters, 1 byte per subspace
          case BQ -> 1;
          case RABITQ -> 1;
          case NVQ -> 8;
        };
    return 32.0 / bitsPerDim;
  }

  private static void addAll(VectorCollection col, float[][] corpus) {
    List<Document> batch = new ArrayList<>(1000);
    for (int i = 0; i < corpus.length; i++) {
      batch.add(Document.of("doc-" + i, corpus[i]));
      if (batch.size() == 1000) {
        col.addAll(batch);
        batch.clear();
      }
    }
    if (!batch.isEmpty()) {
      col.addAll(batch);
    }
  }
}
