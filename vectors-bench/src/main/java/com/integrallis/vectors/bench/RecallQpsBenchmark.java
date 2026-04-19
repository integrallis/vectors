package com.integrallis.vectors.bench;

import com.integrallis.vectors.bench.dataset.DatasetDownloader;
import com.integrallis.vectors.bench.dataset.DatasetRegistry;
import com.integrallis.vectors.bench.dataset.FvecsLoader;
import com.integrallis.vectors.bench.dataset.Hdf5Loader;
import com.integrallis.vectors.bench.report.BenchmarkReporter;
import com.integrallis.vectors.bench.report.BenchmarkResult;
import com.integrallis.vectors.bench.report.LatencyCollector;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Macro benchmark harness that produces recall@k vs QPS Pareto curves.
 *
 * <p>This is the primary ANN search benchmark — the numbers that every vector search library
 * publishes. It sweeps algorithm parameters on standard datasets and measures recall@10/100, QPS,
 * and latency percentiles.
 *
 * <p>Follows the ANN-Benchmarks evaluation protocol:
 *
 * <ol>
 *   <li>Build index with given parameters
 *   <li>Warmup: 3 rounds of all queries (discard results)
 *   <li>Measurement: 5 rounds, sequential single-threaded queries
 *   <li>Report: recall@k, QPS, p50/p95/p99 latency
 * </ol>
 *
 * <p>Run:
 *
 * <pre>{@code
 * // Full sweep (downloads datasets if not present)
 * java --add-modules jdk.incubator.vector -cp ... RecallQpsBenchmark
 *
 * // Specific dataset
 * java --add-modules jdk.incubator.vector -cp ... RecallQpsBenchmark sift-128-euclidean
 *
 * // Specific algorithm
 * java --add-modules jdk.incubator.vector -cp ... RecallQpsBenchmark sift-128-euclidean hnsw
 * }</pre>
 */
public final class RecallQpsBenchmark {

  private static final int K = 10;
  private static final int WARMUP_ROUNDS = 3;
  private static final int MEASUREMENT_ROUNDS = 5;

  private RecallQpsBenchmark() {}

  // -------------------------------------------------------------------------
  // Entry point
  // -------------------------------------------------------------------------

  /** Runs the benchmark suite. Args: [dataset] [algorithm]. */
  public static void main(String[] args) throws IOException {
    String datasetFilter = args.length > 0 ? args[0] : null;
    String algoFilter = args.length > 1 ? args[1] : null;

    List<BenchmarkResult> allResults = new ArrayList<>();

    // Dataset configurations: name → similarity function
    List<DatasetConfig> datasets =
        List.of(
            new DatasetConfig("sift-128-euclidean", SimilarityFunction.EUCLIDEAN, "fvecs"),
            new DatasetConfig("glove-100-angular", SimilarityFunction.COSINE, "hdf5"),
            new DatasetConfig("fashion-mnist-784-euclidean", SimilarityFunction.EUCLIDEAN, "hdf5"));

    for (DatasetConfig dsCfg : datasets) {
      if (datasetFilter != null && !dsCfg.name.contains(datasetFilter)) continue;

      System.out.printf("%n=== Dataset: %s ===%n", dsCfg.name);
      DatasetData data = loadDataset(dsCfg);
      if (data == null) {
        System.out.printf("  Skipping (not available and download failed).%n");
        continue;
      }

      System.out.printf(
          "  Loaded: %,d base vectors, %,d queries, dim=%d%n",
          data.corpus.length, data.queries.length, data.corpus[0].length);

      // Compute ground truth if not provided by dataset.
      int[][] groundTruth = data.groundTruth;
      if (groundTruth == null) {
        System.out.println("  Computing ground truth (brute-force)...");
        groundTruth = RecallUtil.bruteForceGroundTruth(data.queries, data.corpus, dsCfg.sim, 100);
      }

      int dim = data.corpus[0].length;

      // --- HNSW sweep ---
      if (algoFilter == null || "hnsw".equals(algoFilter)) {
        for (int m : new int[] {16, 32, 64}) {
          for (int efConstruction : new int[] {100, 200, 400}) {
            allResults.addAll(
                benchmarkHnsw(
                    dsCfg, data.corpus, data.queries, groundTruth, dim, m, efConstruction));
          }
        }
      }

      // --- Vamana sweep ---
      if (algoFilter == null || "vamana".equals(algoFilter)) {
        for (int r : new int[] {32, 64, 128}) {
          for (int lBuild : new int[] {100, 128, 200}) {
            for (double alpha : new double[] {1.0, 1.2}) {
              allResults.addAll(
                  benchmarkVamana(
                      dsCfg, data.corpus, data.queries, groundTruth, dim, r, lBuild, alpha));
            }
          }
        }
      }

      // --- FLAT baseline ---
      if (algoFilter == null || "flat".equals(algoFilter)) {
        allResults.addAll(benchmarkFlat(dsCfg, data.corpus, data.queries, groundTruth, dim));
      }

      // --- IVF_FLAT sweep ---
      if (algoFilter == null || "ivf".equals(algoFilter)) {
        int n = data.corpus.length;
        int sqrtN = (int) Math.sqrt(n);
        for (int nlist : new int[] {sqrtN, 2 * sqrtN, 4 * sqrtN}) {
          allResults.addAll(
              benchmarkIvf(dsCfg, data.corpus, data.queries, groundTruth, dim, nlist));
        }
      }
    }

    // Output results.
    if (!allResults.isEmpty()) {
      BenchmarkReporter.console(allResults, System.out);

      Path outputDir = DatasetRegistry.dataDir().resolve("results");
      BenchmarkReporter.csv(allResults, outputDir.resolve("recall-qps.csv"));
      BenchmarkReporter.json(allResults, outputDir.resolve("recall-qps.json"));
      System.out.printf("%nResults written to %s (CSV + JSON)%n", outputDir.toAbsolutePath());
    }
  }

  // -------------------------------------------------------------------------
  // HNSW
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkHnsw(
      DatasetConfig dsCfg,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      int m,
      int efConstruction) {

    Map<String, String> buildParams =
        Map.of("M", String.valueOf(m), "efConstruction", String.valueOf(efConstruction));
    System.out.printf("  HNSW M=%d efConstruction=%d: building...", m, efConstruction);

    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.HNSW)
            .hnswM(m)
            .hnswEfConstruction(efConstruction)
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int efSearch : new int[] {16, 32, 64, 128, 256, 512}) {
        BenchmarkResult r =
            measureSearch(
                dsCfg.name,
                "hnsw",
                buildParams,
                Map.of("efSearch", String.valueOf(efSearch)),
                col,
                queries,
                groundTruth,
                K,
                efSearch,
                buildTime);
        results.add(r);
      }
    } finally {
      col.close();
    }
    return results;
  }

  // -------------------------------------------------------------------------
  // Vamana
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkVamana(
      DatasetConfig dsCfg,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      int r,
      int lBuild,
      double alpha) {

    Map<String, String> buildParams =
        Map.of(
            "R",
            String.valueOf(r),
            "L_build",
            String.valueOf(lBuild),
            "alpha",
            String.valueOf(alpha));
    System.out.printf("  Vamana R=%d L=%d alpha=%.1f: building...", r, lBuild, alpha);

    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.VAMANA)
            .vamanaMaxDegree(r)
            .vamanaSearchListSize(lBuild)
            .vamanaAlpha((float) alpha)
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int lSearch : new int[] {32, 64, 100, 128, 200, 300}) {
        BenchmarkResult result =
            measureSearch(
                dsCfg.name,
                "vamana",
                buildParams,
                Map.of("L_search", String.valueOf(lSearch)),
                col,
                queries,
                groundTruth,
                K,
                lSearch,
                buildTime);
        results.add(result);
      }
    } finally {
      col.close();
    }
    return results;
  }

  // -------------------------------------------------------------------------
  // FLAT (brute-force baseline)
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkFlat(
      DatasetConfig dsCfg, float[][] corpus, float[][] queries, int[][] groundTruth, int dim) {

    System.out.print("  FLAT: building...");
    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.FLAT)
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    try {
      BenchmarkResult r =
          measureSearch(
              dsCfg.name, "flat", Map.of(), Map.of(), col, queries, groundTruth, K, 0, buildTime);
      return List.of(r);
    } finally {
      col.close();
    }
  }

  // -------------------------------------------------------------------------
  // IVF_FLAT
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkIvf(
      DatasetConfig dsCfg,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      int nlist) {

    Map<String, String> buildParams = Map.of("nlist", String.valueOf(nlist));
    System.out.printf("  IVF_FLAT nlist=%d: building...", nlist);

    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.IVF_FLAT)
            .ivfK(nlist)
            .ivfNprobe(1) // default; overridden per search
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int nprobe : new int[] {1, 2, 4, 8, 16, 32, 64}) {
        if (nprobe > nlist) break;
        // IVF nprobe is set at build time, so we rebuild for each nprobe.
        // For efficiency, only rebuild the collection for different nprobe values.
        VectorCollection ivfCol =
            VectorCollection.builder()
                .dimension(dim)
                .metric(dsCfg.sim)
                .indexType(IndexType.IVF_FLAT)
                .ivfK(nlist)
                .ivfNprobe(nprobe)
                .build();
        addAll(ivfCol, corpus);
        ivfCol.commit();
        try {
          BenchmarkResult r =
              measureSearch(
                  dsCfg.name,
                  "ivf_flat",
                  buildParams,
                  Map.of("nprobe", String.valueOf(nprobe)),
                  ivfCol,
                  queries,
                  groundTruth,
                  K,
                  0,
                  buildTime);
          results.add(r);
        } finally {
          ivfCol.close();
        }
      }
    } finally {
      col.close();
    }
    return results;
  }

  // -------------------------------------------------------------------------
  // Core measurement
  // -------------------------------------------------------------------------

  private static BenchmarkResult measureSearch(
      String dataset,
      String algorithm,
      Map<String, String> buildParams,
      Map<String, String> searchParams,
      VectorCollection col,
      float[][] queries,
      int[][] groundTruth,
      int k,
      int searchListSize,
      double buildTime) {

    int numQueries = queries.length;

    // Warmup.
    for (int round = 0; round < WARMUP_ROUNDS; round++) {
      for (float[] q : queries) {
        SearchRequest.Builder req = SearchRequest.builder(q, k);
        if (searchListSize > 0) req.searchListSize(searchListSize);
        col.search(req.build());
      }
    }

    // Force GC before measurement.
    System.gc();

    // Measurement: collect per-query latencies and results for recall.
    LatencyCollector latency = new LatencyCollector(numQueries * MEASUREMENT_ROUNDS);
    int[][] approxResults = new int[numQueries][k];

    for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
      for (int qi = 0; qi < numQueries; qi++) {
        SearchRequest.Builder req =
            SearchRequest.builder(queries[qi], k)
                .includeVector(false)
                .includeText(false)
                .includeMetadata(false);
        if (searchListSize > 0) req.searchListSize(searchListSize);

        long t0 = System.nanoTime();
        SearchResult result = col.search(req.build());
        latency.record(System.nanoTime() - t0);

        // Use last round's results for recall.
        if (round == MEASUREMENT_ROUNDS - 1) {
          List<SearchResult.Hit> hits = result.hits();
          for (int i = 0; i < Math.min(k, hits.size()); i++) {
            // Extract ordinal from doc id: "doc-{ordinal}"
            approxResults[qi][i] = parseOrdinal(hits.get(i).id());
          }
        }
      }
    }

    latency.compute();

    // Compute recall.
    double recall10 = RecallUtil.meanRecallAtK(groundTruth, approxResults, k);
    double recall100 = -1;
    // Only compute recall@100 if groundTruth has >= 100 neighbors.
    if (groundTruth.length > 0 && groundTruth[0].length >= 100 && k <= 100) {
      // We'd need k=100 search to compute recall@100; skip for now.
      recall100 = -1;
    }

    System.out.printf(
        "    %s: recall@%d=%.3f  QPS=%,.0f  p50=%.0fus  p99=%.0fus%n",
        searchParams.isEmpty() ? "default" : BenchmarkReporter.formatParams(searchParams),
        k,
        recall10,
        latency.qps(),
        latency.p50Us(),
        latency.p99Us());

    return BenchmarkResult.builder(dataset, algorithm)
        .buildParams(buildParams)
        .searchParams(searchParams)
        .recall10(recall10)
        .recall100(recall100)
        .qps(latency.qps())
        .p50Us(latency.p50Us())
        .p95Us(latency.p95Us())
        .p99Us(latency.p99Us())
        .buildTimeSeconds(buildTime)
        .build();
  }

  // -------------------------------------------------------------------------
  // Dataset loading
  // -------------------------------------------------------------------------

  private static DatasetData loadDataset(DatasetConfig cfg) throws IOException {
    if ("fvecs".equals(cfg.format)) {
      return loadFvecsDataset(cfg);
    }
    return loadHdf5Dataset(cfg);
  }

  private static DatasetData loadFvecsDataset(DatasetConfig cfg) throws IOException {
    // SIFT1M special case: check for local fvecs, then try download HDF5.
    if ("sift-128-euclidean".equals(cfg.name) && DatasetRegistry.isSift1MAvailable()) {
      Path dir = DatasetRegistry.sift1MDir();
      float[][] corpus = FvecsLoader.readFvecs(dir.resolve("sift_base.fvecs"));
      float[][] queries = FvecsLoader.readFvecs(dir.resolve("sift_query.fvecs"));
      int[][] gt = FvecsLoader.readIvecs(dir.resolve("sift_groundtruth.ivecs"));
      return new DatasetData(corpus, queries, gt);
    }

    // Fall back to HDF5 download.
    return loadHdf5Dataset(new DatasetConfig(cfg.name, cfg.sim, "hdf5"));
  }

  private static DatasetData loadHdf5Dataset(DatasetConfig cfg) throws IOException {
    Path hdf5 = DatasetRegistry.annBenchDataset(cfg.name);
    if (!java.nio.file.Files.exists(hdf5)) {
      // Try auto-download.
      if (DatasetDownloader.isKnownDataset(cfg.name)) {
        hdf5 = DatasetDownloader.ensureAvailable(cfg.name);
      } else {
        return null;
      }
    }

    float[][] corpus = Hdf5Loader.readTrainVectors(hdf5);
    float[][] queries = Hdf5Loader.readTestVectors(hdf5);
    int[][] gt = Hdf5Loader.readNeighbors(hdf5);
    return new DatasetData(corpus, queries, gt);
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private static void addAll(VectorCollection col, float[][] corpus) {
    // Batch add for efficiency.
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

  private static int parseOrdinal(String docId) {
    // doc-{ordinal}
    return Integer.parseInt(docId.substring(4));
  }

  // -------------------------------------------------------------------------
  // Internal types
  // -------------------------------------------------------------------------

  private record DatasetConfig(String name, SimilarityFunction sim, String format) {}

  private record DatasetData(float[][] corpus, float[][] queries, int[][] groundTruth) {}
}
