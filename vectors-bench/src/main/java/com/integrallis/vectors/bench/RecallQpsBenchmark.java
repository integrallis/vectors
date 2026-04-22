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
import com.integrallis.vectors.hnsw.HnswFusedAdcIndex;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
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

  /** Dataset-size threshold at which the default profile drops the heaviest HNSW configs. */
  private static final int LARGE_CORPUS_THRESHOLD = 500_000;

  private RecallQpsBenchmark() {}

  // -------------------------------------------------------------------------
  // Entry point
  // -------------------------------------------------------------------------

  /** Runs the benchmark suite. Args: [dataset] [algorithm]. */
  public static void main(String[] args) throws IOException {
    String datasetFilter = args.length > 0 ? args[0] : null;
    String algoFilter = args.length > 1 ? args[1] : null;

    Path outputDir = DatasetRegistry.dataDir().resolve("results");
    Path csvFile = outputDir.resolve("recall-qps.csv");
    Path jsonFile = outputDir.resolve("recall-qps.json");
    Path markersDir = outputDir.resolve("recall-qps.markers");
    CheckpointStore checkpoint = new CheckpointStore(csvFile, markersDir);

    String profile = System.getProperty("bench.profile", "default");
    int hnswThreads =
        Integer.getInteger(
            "bench.hnsw.threads", Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
    int vamanaThreads =
        Integer.getInteger(
            "bench.vamana.threads", Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
    System.out.printf(
        "[recallQps] profile=%s hnsw.threads=%d vamana.threads=%d checkpoint=%s%n",
        profile, hnswThreads, vamanaThreads, csvFile.toAbsolutePath());

    List<BenchmarkResult> sessionResults = new ArrayList<>();

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
      int n = data.corpus.length;
      SweepConfig sweep = SweepConfig.resolve(profile, n);
      System.out.printf(
          "  sweep: HNSW M=%s efC=%s efSearch=%s | Vamana R=%s L=%s alpha=%s lSearch=%s |"
              + " IVF nprobe=%s%n",
          Arrays.toString(sweep.hnswM),
          Arrays.toString(sweep.hnswEf),
          Arrays.toString(sweep.hnswEfSearch),
          Arrays.toString(sweep.vamanaR),
          Arrays.toString(sweep.vamanaL),
          Arrays.toString(sweep.vamanaAlpha),
          Arrays.toString(sweep.vamanaLSearch),
          Arrays.toString(sweep.ivfNprobe));

      // --- HNSW sweep ---
      if (algoFilter == null || "hnsw".equals(algoFilter)) {
        for (int m : sweep.hnswM) {
          for (int efConstruction : sweep.hnswEf) {
            sessionResults.addAll(
                benchmarkHnsw(
                    dsCfg,
                    data.corpus,
                    data.queries,
                    groundTruth,
                    dim,
                    m,
                    efConstruction,
                    sweep.hnswEfSearch,
                    hnswThreads,
                    checkpoint,
                    csvFile));
          }
        }
      }

      // --- HNSW + Fused ADC two-pass sweep (PQ-compressed beam, full-precision rerank) ---
      if ("hnsw_fused_adc".equals(algoFilter)) {
        // When -Pbench.adc.pq is not specified, fall back to HnswFusedAdcIndex.defaultSubvectors
        // (dim/8 clamped to the nearest divisor), which lands recall near 0.997 without tuning.
        int pqTarget = Integer.getInteger("bench.adc.pq", HnswFusedAdcIndex.defaultSubvectors(dim));
        int pqSubvectors = bestPqSubvectors(dim, pqTarget);
        int pqClusters = Integer.getInteger("bench.adc.clusters", 256);
        float[] overQueryValues = parseFloats(System.getProperty("bench.adc.overQuery"), 2.0f);
        float anisoThreshold = Float.parseFloat(System.getProperty("bench.adc.aniso", "-1.0"));
        for (int m : sweep.hnswM) {
          for (int efConstruction : sweep.hnswEf) {
            sessionResults.addAll(
                benchmarkHnswFusedAdc(
                    dsCfg,
                    data.corpus,
                    data.queries,
                    groundTruth,
                    dim,
                    m,
                    efConstruction,
                    sweep.hnswEfSearch,
                    pqSubvectors,
                    pqClusters,
                    overQueryValues,
                    anisoThreshold,
                    checkpoint,
                    csvFile));
          }
        }
      }

      // --- Vamana sweep ---
      if (algoFilter == null || "vamana".equals(algoFilter)) {
        for (int r : sweep.vamanaR) {
          for (int lBuild : sweep.vamanaL) {
            for (double alpha : sweep.vamanaAlpha) {
              sessionResults.addAll(
                  benchmarkVamana(
                      dsCfg,
                      data.corpus,
                      data.queries,
                      groundTruth,
                      dim,
                      r,
                      lBuild,
                      alpha,
                      sweep.vamanaLSearch,
                      vamanaThreads,
                      checkpoint,
                      csvFile));
            }
          }
        }
      }

      // --- FLAT baseline ---
      if (algoFilter == null || "flat".equals(algoFilter)) {
        sessionResults.addAll(
            benchmarkFlat(dsCfg, data.corpus, data.queries, groundTruth, dim, checkpoint, csvFile));
      }

      // --- IVF_FLAT sweep ---
      if (algoFilter == null || "ivf".equals(algoFilter)) {
        int sqrtN = (int) Math.sqrt(n);
        for (int nlist : new int[] {sqrtN, 2 * sqrtN, 4 * sqrtN}) {
          sessionResults.addAll(
              benchmarkIvf(
                  dsCfg,
                  data.corpus,
                  data.queries,
                  groundTruth,
                  dim,
                  nlist,
                  sweep.ivfNprobe,
                  checkpoint,
                  csvFile));
        }
      }
    }

    // Final output: console table for this session's results, JSON snapshot of this session.
    // CSV is the cumulative source of truth — rows have already been appended incrementally.
    if (!sessionResults.isEmpty()) {
      BenchmarkReporter.console(sessionResults, System.out);
      BenchmarkReporter.json(sessionResults, jsonFile);
      System.out.printf(
          "%nResults: CSV=%s%n         JSON (this session)=%s%n",
          csvFile.toAbsolutePath(), jsonFile.toAbsolutePath());
    } else {
      System.out.printf(
          "%nAll configurations were already completed. CSV=%s%n", csvFile.toAbsolutePath());
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
      int efConstruction,
      int[] efSearchValues,
      int hnswThreads,
      CheckpointStore checkpoint,
      Path csvFile) {

    Map<String, String> buildParams =
        Map.of("M", String.valueOf(m), "efConstruction", String.valueOf(efConstruction));
    if (isFullyCompleted(checkpoint, dsCfg.name, "hnsw", buildParams, efSearchValues, "efSearch")) {
      System.out.printf(
          "  HNSW M=%d efConstruction=%d: already completed, skipping%n", m, efConstruction);
      checkpoint.markBuildCompleted(dsCfg.name, "hnsw", buildParams);
      return List.of();
    }
    System.out.printf(
        "  HNSW M=%d efConstruction=%d threads=%d: building...", m, efConstruction, hnswThreads);

    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.HNSW)
            .hnswM(m)
            .hnswEfConstruction(efConstruction)
            .hnswBuildThreads(hnswThreads)
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int efSearch : efSearchValues) {
        Map<String, String> searchParams = Map.of("efSearch", String.valueOf(efSearch));
        if (checkpoint.isRowCompleted(dsCfg.name, "hnsw", buildParams, searchParams)) {
          System.out.printf("    efSearch=%d: cached, skipping%n", efSearch);
          continue;
        }
        BenchmarkResult r =
            measureSearch(
                dsCfg.name,
                "hnsw",
                buildParams,
                searchParams,
                col,
                queries,
                groundTruth,
                K,
                efSearch,
                buildTime);
        recordResult(r, checkpoint, csvFile);
        results.add(r);
      }
    } finally {
      col.close();
    }
    checkpoint.markBuildCompleted(dsCfg.name, "hnsw", buildParams);
    return results;
  }

  // -------------------------------------------------------------------------
  // HNSW + Fused ADC (PQ-compressed beam + exact rerank)
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkHnswFusedAdc(
      DatasetConfig dsCfg,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      int m,
      int efConstruction,
      int[] efSearchValues,
      int pqSubvectors,
      int pqClusters,
      float[] overQueryValues,
      float anisoThreshold,
      CheckpointStore checkpoint,
      Path csvFile) {

    // Include anisoThreshold in buildParams only when non-default so pre-Phase-D rows keep their
    // original key and don't become "missing" on the old checkpoint.
    Map<String, String> buildParams;
    if (anisoThreshold >= 0f) {
      buildParams =
          Map.of(
              "M", String.valueOf(m),
              "efConstruction", String.valueOf(efConstruction),
              "pqSubvectors", String.valueOf(pqSubvectors),
              "pqClusters", String.valueOf(pqClusters),
              "aniso", String.valueOf(anisoThreshold));
    } else {
      buildParams =
          Map.of(
              "M", String.valueOf(m),
              "efConstruction", String.valueOf(efConstruction),
              "pqSubvectors", String.valueOf(pqSubvectors),
              "pqClusters", String.valueOf(pqClusters));
    }

    // Pre-compute the full efSearch x overQuery search grid and skip the build entirely if every
    // combination is already recorded in the checkpoint.
    boolean anyPending = false;
    for (int efSearch : efSearchValues) {
      for (float oq : overQueryValues) {
        Map<String, String> sp =
            Map.of("efSearch", String.valueOf(efSearch), "overQuery", String.valueOf(oq));
        if (!checkpoint.isRowCompleted(dsCfg.name, "hnsw_fused_adc", buildParams, sp)) {
          anyPending = true;
          break;
        }
      }
      if (anyPending) break;
    }
    if (!anyPending) {
      System.out.printf(
          "  HNSW+FusedADC M=%d efC=%d pq=%d/%d: already completed, skipping%n",
          m, efConstruction, pqSubvectors, pqClusters);
      checkpoint.markBuildCompleted(dsCfg.name, "hnsw_fused_adc", buildParams);
      return List.of();
    }
    System.out.printf(
        "  HNSW+FusedADC M=%d efC=%d pq=%d/%d%s: building...",
        m,
        efConstruction,
        pqSubvectors,
        pqClusters,
        anisoThreshold >= 0f ? String.format(" aniso=%.2f", anisoThreshold) : "");

    long t0 = System.nanoTime();
    HnswFusedAdcIndex idx =
        HnswFusedAdcIndex.build(
            corpus, dsCfg.sim, m, efConstruction, pqSubvectors, pqClusters, 42L, anisoThreshold);
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    for (float overQuery : overQueryValues) {
      for (int efSearch : efSearchValues) {
        Map<String, String> searchParams =
            Map.of("efSearch", String.valueOf(efSearch), "overQuery", String.valueOf(overQuery));
        if (checkpoint.isRowCompleted(dsCfg.name, "hnsw_fused_adc", buildParams, searchParams)) {
          System.out.printf(
              "    efSearch=%d overQuery=%.1f: cached, skipping%n", efSearch, overQuery);
          continue;
        }
        BenchmarkResult r =
            measureSearchFusedAdc(
                dsCfg.name,
                buildParams,
                searchParams,
                idx,
                queries,
                groundTruth,
                K,
                efSearch,
                overQuery,
                buildTime);
        recordResult(r, checkpoint, csvFile);
        results.add(r);
      }
    }
    checkpoint.markBuildCompleted(dsCfg.name, "hnsw_fused_adc", buildParams);
    return results;
  }

  /**
   * Measures {@link HnswFusedAdcIndex#searchTwoPass} directly — unlike {@link #measureSearch} it
   * bypasses the {@link VectorCollection} facade because the Fused ADC index isn't wired into the
   * standard {@code IndexType} enum.
   */
  private static BenchmarkResult measureSearchFusedAdc(
      String dataset,
      Map<String, String> buildParams,
      Map<String, String> searchParams,
      HnswFusedAdcIndex idx,
      float[][] queries,
      int[][] groundTruth,
      int k,
      int efSearch,
      float overQuery,
      double buildTime) {

    int numQueries = queries.length;

    for (int round = 0; round < WARMUP_ROUNDS; round++) {
      for (float[] q : queries) idx.searchTwoPass(q, k, efSearch, overQuery);
    }
    System.gc();

    LatencyCollector latency = new LatencyCollector(numQueries * MEASUREMENT_ROUNDS);
    int[][] approxResults = new int[numQueries][k];
    for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
      for (int qi = 0; qi < numQueries; qi++) {
        long t0 = System.nanoTime();
        com.integrallis.vectors.hnsw.SearchResult r =
            idx.searchTwoPass(queries[qi], k, efSearch, overQuery);
        latency.record(System.nanoTime() - t0);
        if (round == MEASUREMENT_ROUNDS - 1) {
          int[] ids = r.nodeIds();
          int take = Math.min(k, ids.length);
          System.arraycopy(ids, 0, approxResults[qi], 0, take);
          for (int i = take; i < k; i++) approxResults[qi][i] = -1;
        }
      }
    }
    latency.compute();
    double recall10 = RecallUtil.meanRecallAtK(groundTruth, approxResults, k);

    System.out.printf(
        "    %s: recall@%d=%.3f  QPS=%,.0f  p50=%.0fus  p99=%.0fus%n",
        BenchmarkReporter.formatParams(searchParams),
        k,
        recall10,
        latency.qps(),
        latency.p50Us(),
        latency.p99Us());

    return BenchmarkResult.builder(dataset, "hnsw_fused_adc")
        .buildParams(buildParams)
        .searchParams(searchParams)
        .recall10(recall10)
        .recall100(-1)
        .qps(latency.qps())
        .p50Us(latency.p50Us())
        .p95Us(latency.p95Us())
        .p99Us(latency.p99Us())
        .buildTimeSeconds(buildTime)
        .build();
  }

  /**
   * Returns the divisor of {@code dim} closest to {@code target} (ties broken toward the larger
   * divisor). {@link HnswFusedAdcIndex#build} requires {@code pqSubvectors} to divide the embedding
   * dimension exactly — this helper picks a practical default.
   */
  private static int bestPqSubvectors(int dim, int target) {
    int best = 1;
    int bestDist = Math.abs(1 - target);
    for (int k = 2; k <= dim; k++) {
      if (dim % k != 0) continue;
      int d = Math.abs(k - target);
      if (d < bestDist || (d == bestDist && k > best)) {
        best = k;
        bestDist = d;
      }
    }
    return best;
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
      double alpha,
      int[] lSearchValues,
      int buildThreads,
      CheckpointStore checkpoint,
      Path csvFile) {

    Map<String, String> buildParams =
        Map.of(
            "R", String.valueOf(r),
            "L_build", String.valueOf(lBuild),
            "alpha", String.valueOf(alpha));
    if (isFullyCompleted(
        checkpoint, dsCfg.name, "vamana", buildParams, lSearchValues, "L_search")) {
      System.out.printf(
          "  Vamana R=%d L=%d alpha=%.1f: already completed, skipping%n", r, lBuild, alpha);
      checkpoint.markBuildCompleted(dsCfg.name, "vamana", buildParams);
      return List.of();
    }
    System.out.printf(
        "  Vamana R=%d L=%d alpha=%.1f threads=%d: building...", r, lBuild, alpha, buildThreads);

    long t0 = System.nanoTime();
    VectorCollection col =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.VAMANA)
            .vamanaMaxDegree(r)
            .vamanaSearchListSize(lBuild)
            .vamanaAlpha((float) alpha)
            .vamanaBuildThreads(buildThreads)
            .build();
    addAll(col, corpus);
    col.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int lSearch : lSearchValues) {
        Map<String, String> searchParams = Map.of("L_search", String.valueOf(lSearch));
        if (checkpoint.isRowCompleted(dsCfg.name, "vamana", buildParams, searchParams)) {
          System.out.printf("    L_search=%d: cached, skipping%n", lSearch);
          continue;
        }
        BenchmarkResult result =
            measureSearch(
                dsCfg.name,
                "vamana",
                buildParams,
                searchParams,
                col,
                queries,
                groundTruth,
                K,
                lSearch,
                buildTime);
        recordResult(result, checkpoint, csvFile);
        results.add(result);
      }
    } finally {
      col.close();
    }
    checkpoint.markBuildCompleted(dsCfg.name, "vamana", buildParams);
    return results;
  }

  // -------------------------------------------------------------------------
  // FLAT (brute-force baseline)
  // -------------------------------------------------------------------------

  private static List<BenchmarkResult> benchmarkFlat(
      DatasetConfig dsCfg,
      float[][] corpus,
      float[][] queries,
      int[][] groundTruth,
      int dim,
      CheckpointStore checkpoint,
      Path csvFile) {

    Map<String, String> buildParams = Map.of();
    Map<String, String> searchParams = Map.of();
    if (checkpoint.isRowCompleted(dsCfg.name, "flat", buildParams, searchParams)) {
      System.out.println("  FLAT: already completed, skipping");
      checkpoint.markBuildCompleted(dsCfg.name, "flat", buildParams);
      return List.of();
    }

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
              dsCfg.name,
              "flat",
              buildParams,
              searchParams,
              col,
              queries,
              groundTruth,
              K,
              0,
              buildTime);
      recordResult(r, checkpoint, csvFile);
      checkpoint.markBuildCompleted(dsCfg.name, "flat", buildParams);
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
      int nlist,
      int[] nprobeValues,
      CheckpointStore checkpoint,
      Path csvFile) {

    Map<String, String> buildParams = Map.of("nlist", String.valueOf(nlist));
    // Filter out nprobe values that exceed nlist before computing the "fully completed" check.
    int[] effectiveNprobe = filterAtMost(nprobeValues, nlist);
    if (effectiveNprobe.length == 0) return List.of();
    if (isFullyCompleted(
        checkpoint, dsCfg.name, "ivf_flat", buildParams, effectiveNprobe, "nprobe")) {
      System.out.printf("  IVF_FLAT nlist=%d: already completed, skipping%n", nlist);
      checkpoint.markBuildCompleted(dsCfg.name, "ivf_flat", buildParams);
      return List.of();
    }

    // Skip nprobe values already measured for this nlist; build-once-search-many means we only
    // need to build the index if at least one nprobe in the sweep is not yet checkpointed.
    int[] pendingNprobe = new int[effectiveNprobe.length];
    int pendingCount = 0;
    for (int nprobe : effectiveNprobe) {
      Map<String, String> searchParams = Map.of("nprobe", String.valueOf(nprobe));
      if (checkpoint.isRowCompleted(dsCfg.name, "ivf_flat", buildParams, searchParams)) {
        System.out.printf("  IVF_FLAT nlist=%d nprobe=%d: cached, skipping%n", nlist, nprobe);
      } else {
        pendingNprobe[pendingCount++] = nprobe;
      }
    }
    if (pendingCount == 0) {
      checkpoint.markBuildCompleted(dsCfg.name, "ivf_flat", buildParams);
      return List.of();
    }

    // Single build per nlist: the constructor-time nprobe is used as the default; the per-query
    // nprobe sweep is driven through SearchRequest.searchListSize (honored by IvfFlatAdapter).
    int defaultNprobe = pendingNprobe[0];
    System.out.printf(
        "  IVF_FLAT nlist=%d (sweep %d nprobe values): building...", nlist, pendingCount);
    long t0 = System.nanoTime();
    VectorCollection ivfCol =
        VectorCollection.builder()
            .dimension(dim)
            .metric(dsCfg.sim)
            .indexType(IndexType.IVF_FLAT)
            .ivfK(nlist)
            .ivfNprobe(defaultNprobe)
            .build();
    addAll(ivfCol, corpus);
    ivfCol.commit();
    double buildTime = (System.nanoTime() - t0) / 1e9;
    System.out.printf(" %.1fs%n", buildTime);

    List<BenchmarkResult> results = new ArrayList<>();
    try {
      for (int i = 0; i < pendingCount; i++) {
        int nprobe = pendingNprobe[i];
        Map<String, String> searchParams = Map.of("nprobe", String.valueOf(nprobe));
        BenchmarkResult r =
            measureSearch(
                dsCfg.name,
                "ivf_flat",
                buildParams,
                searchParams,
                ivfCol,
                queries,
                groundTruth,
                K,
                nprobe,
                buildTime);
        recordResult(r, checkpoint, csvFile);
        results.add(r);
      }
    } finally {
      ivfCol.close();
    }
    checkpoint.markBuildCompleted(dsCfg.name, "ivf_flat", buildParams);
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
          int take = Math.min(k, hits.size());
          for (int i = 0; i < take; i++) {
            // Extract ordinal from doc id: "doc-{ordinal}"
            approxResults[qi][i] = parseOrdinal(hits.get(i).id());
          }
          // Fill unused slots with -1 so they cannot accidentally match ordinal 0 in ground truth.
          for (int i = take; i < k; i++) approxResults[qi][i] = -1;
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
  // Checkpointing helpers
  // -------------------------------------------------------------------------

  private static boolean isFullyCompleted(
      CheckpointStore checkpoint,
      String dataset,
      String algorithm,
      Map<String, String> buildParams,
      int[] searchValues,
      String searchParamKey) {
    if (checkpoint.isBuildCompleted(dataset, algorithm, buildParams)) return true;
    for (int v : searchValues) {
      if (!checkpoint.isRowCompleted(
          dataset, algorithm, buildParams, Map.of(searchParamKey, String.valueOf(v)))) {
        return false;
      }
    }
    return true;
  }

  private static void recordResult(
      BenchmarkResult result, CheckpointStore checkpoint, Path csvFile) {
    BenchmarkReporter.csvAppend(result, csvFile);
    checkpoint.markRowCompleted(
        result.dataset(), result.algorithm(), result.buildParams(), result.searchParams());
  }

  private static int[] filterAtMost(int[] values, int cap) {
    int count = 0;
    for (int v : values) if (v <= cap) count++;
    int[] out = new int[count];
    int j = 0;
    for (int v : values) if (v <= cap) out[j++] = v;
    return out;
  }

  private static float[] parseFloats(String csv, float fallback) {
    if (csv == null || csv.isBlank()) return new float[] {fallback};
    String[] parts = csv.split(",");
    float[] out = new float[parts.length];
    for (int i = 0; i < parts.length; i++) out[i] = Float.parseFloat(parts[i].trim());
    return out;
  }

  // -------------------------------------------------------------------------
  // Sweep configuration
  // -------------------------------------------------------------------------

  /**
   * Per-dataset sweep parameters. Resolved from the {@code bench.profile} system property
   * (quick|default|full) with optional per-axis overrides via {@code bench.<algo>.<param>} comma
   * lists.
   */
  private static final class SweepConfig {
    final int[] hnswM;
    final int[] hnswEf;
    final int[] hnswEfSearch;
    final int[] vamanaR;
    final int[] vamanaL;
    final double[] vamanaAlpha;
    final int[] vamanaLSearch;
    final int[] ivfNprobe;

    private SweepConfig(
        int[] hnswM,
        int[] hnswEf,
        int[] hnswEfSearch,
        int[] vamanaR,
        int[] vamanaL,
        double[] vamanaAlpha,
        int[] vamanaLSearch,
        int[] ivfNprobe) {
      this.hnswM = hnswM;
      this.hnswEf = hnswEf;
      this.hnswEfSearch = hnswEfSearch;
      this.vamanaR = vamanaR;
      this.vamanaL = vamanaL;
      this.vamanaAlpha = vamanaAlpha;
      this.vamanaLSearch = vamanaLSearch;
      this.ivfNprobe = ivfNprobe;
    }

    static SweepConfig resolve(String profile, int n) {
      // Base grids per profile.
      int[] hnswM, hnswEf, hnswEfSearch;
      int[] vamanaR, vamanaL, vamanaLSearch;
      double[] vamanaAlpha;
      int[] ivfNprobe;
      switch (profile) {
        case "quick" -> {
          hnswM = new int[] {16};
          hnswEf = new int[] {100};
          hnswEfSearch = new int[] {64, 128, 256};
          vamanaR = new int[] {64};
          vamanaL = new int[] {100};
          vamanaAlpha = new double[] {1.2};
          vamanaLSearch = new int[] {64, 128};
          ivfNprobe = new int[] {1, 4, 16};
        }
        case "full" -> {
          hnswM = new int[] {16, 32, 64};
          hnswEf = new int[] {100, 200, 400};
          hnswEfSearch = new int[] {16, 32, 64, 128, 256, 512};
          vamanaR = new int[] {32, 64, 128};
          vamanaL = new int[] {100, 128, 200};
          vamanaAlpha = new double[] {1.0, 1.2};
          vamanaLSearch = new int[] {32, 64, 100, 128, 200, 300};
          ivfNprobe = new int[] {1, 2, 4, 8, 16, 32, 64};
        }
        default -> {
          // "default": full grid, but drop the heaviest HNSW configs for large corpora.
          if (n >= LARGE_CORPUS_THRESHOLD) {
            hnswM = new int[] {16, 32};
            hnswEf = new int[] {100, 200};
          } else {
            hnswM = new int[] {16, 32, 64};
            hnswEf = new int[] {100, 200, 400};
          }
          hnswEfSearch = new int[] {16, 32, 64, 128, 256, 512};
          vamanaR = new int[] {32, 64, 128};
          vamanaL = new int[] {100, 128, 200};
          vamanaAlpha = new double[] {1.0, 1.2};
          vamanaLSearch = new int[] {32, 64, 100, 128, 200, 300};
          ivfNprobe = new int[] {1, 2, 4, 8, 16, 32, 64};
        }
      }
      // Per-axis overrides (comma-separated system properties).
      hnswM = overrideInts("bench.hnsw.m", hnswM);
      hnswEf = overrideInts("bench.hnsw.ef", hnswEf);
      hnswEfSearch = overrideInts("bench.hnsw.efSearch", hnswEfSearch);
      vamanaR = overrideInts("bench.vamana.r", vamanaR);
      vamanaL = overrideInts("bench.vamana.l", vamanaL);
      vamanaAlpha = overrideDoubles("bench.vamana.alpha", vamanaAlpha);
      vamanaLSearch = overrideInts("bench.vamana.lSearch", vamanaLSearch);
      ivfNprobe = overrideInts("bench.ivf.nprobe", ivfNprobe);
      return new SweepConfig(
          hnswM, hnswEf, hnswEfSearch, vamanaR, vamanaL, vamanaAlpha, vamanaLSearch, ivfNprobe);
    }

    private static int[] overrideInts(String key, int[] defaults) {
      String v = System.getProperty(key);
      if (v == null || v.isBlank()) return defaults;
      String[] parts = v.split(",");
      int[] out = new int[parts.length];
      for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i].trim());
      return out;
    }

    private static double[] overrideDoubles(String key, double[] defaults) {
      String v = System.getProperty(key);
      if (v == null || v.isBlank()) return defaults;
      String[] parts = v.split(",");
      double[] out = new double[parts.length];
      for (int i = 0; i < parts.length; i++) out[i] = Double.parseDouble(parts[i].trim());
      return out;
    }
  }

  // -------------------------------------------------------------------------
  // Internal types
  // -------------------------------------------------------------------------

  private record DatasetConfig(String name, SimilarityFunction sim, String format) {}

  private record DatasetData(float[][] corpus, float[][] queries, int[][] groundTruth) {}
}
