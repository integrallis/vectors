package com.integrallis.vectors.bench;

import com.integrallis.vectors.bench.report.BenchmarkReporter;
import com.integrallis.vectors.bench.report.BenchmarkResult;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Macro benchmark measuring index construction scalability.
 *
 * <p>Sweeps corpus size N = {10K, 50K, 100K, 500K, 1M} and reports build time, memory usage, and
 * throughput (vectors/sec) for HNSW and Vamana graph builders.
 *
 * <p>Run:
 *
 * <pre>{@code
 * java --add-modules jdk.incubator.vector -Xmx8g -cp ... BuildScalabilityBenchmark [dim]
 * }</pre>
 */
public final class BuildScalabilityBenchmark {

  private static final long SEED = 42L;

  private BuildScalabilityBenchmark() {}

  /** Runs the scalability benchmark. Args: [dim]. */
  public static void main(String[] args) {
    int dim = args.length > 0 ? Integer.parseInt(args[0]) : 128;

    int[] sizes = {10_000, 50_000, 100_000, 500_000, 1_000_000};
    List<BenchmarkResult> results = new ArrayList<>();
    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();

    System.out.printf("=== Build Scalability Benchmark (dim=%d) ===%n", dim);

    for (int n : sizes) {
      System.out.printf("%n--- N = %,d ---%n", n);
      float[][] corpus = generateCorpus(n, dim);

      // HNSW (M=32, efConstruction=200)
      results.add(
          buildAndMeasure(
              "hnsw",
              IndexType.HNSW,
              corpus,
              dim,
              n,
              Map.of("M", "32", "efConstruction", "200"),
              memBean));

      // Vamana (R=64, L=128, alpha=1.2)
      results.add(
          buildAndMeasure(
              "vamana",
              IndexType.VAMANA,
              corpus,
              dim,
              n,
              Map.of("R", "64", "L", "128", "alpha", "1.2"),
              memBean));

      // FLAT baseline
      results.add(buildAndMeasure("flat", IndexType.FLAT, corpus, dim, n, Map.of(), memBean));

      corpus = null; // Help GC before next size.
      System.gc();
    }

    // Output.
    BenchmarkReporter.console(results, System.out);
    Path outputDir =
        com.integrallis.vectors.bench.dataset.DatasetRegistry.dataDir().resolve("results");
    BenchmarkReporter.csv(results, outputDir.resolve("build-scalability.csv"));
    BenchmarkReporter.json(results, outputDir.resolve("build-scalability.json"));
    System.out.printf("%nResults written to %s%n", outputDir.toAbsolutePath());
  }

  private static BenchmarkResult buildAndMeasure(
      String algorithm,
      IndexType indexType,
      float[][] corpus,
      int dim,
      int n,
      Map<String, String> buildParams,
      MemoryMXBean memBean) {

    System.out.printf("  %s: building %,d vectors...", algorithm.toUpperCase(), n);

    System.gc();
    long heapBefore = memBean.getHeapMemoryUsage().getUsed();

    long t0 = System.nanoTime();
    var builder =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(indexType);

    // Configure algorithm-specific params.
    if (indexType == IndexType.HNSW) {
      builder.hnswM(32).hnswEfConstruction(200);
    } else if (indexType == IndexType.VAMANA) {
      builder.vamanaMaxDegree(64).vamanaSearchListSize(128).vamanaAlpha(1.2f);
    }

    VectorCollection col = builder.build();

    // Batch add.
    List<Document> batch = new ArrayList<>(1000);
    for (int i = 0; i < n; i++) {
      batch.add(Document.of("doc-" + i, corpus[i]));
      if (batch.size() == 1000) {
        col.addAll(batch);
        batch.clear();
      }
    }
    if (!batch.isEmpty()) {
      col.addAll(batch);
    }
    col.commit();

    double buildTimeSec = (System.nanoTime() - t0) / 1e9;
    long heapAfter = memBean.getHeapMemoryUsage().getUsed();
    double memoryMb = Math.max(0, (heapAfter - heapBefore)) / (1024.0 * 1024.0);
    double throughput = n / buildTimeSec;

    col.close();

    System.out.printf(" %.1fs (%,.0f vec/s, %.0f MB)%n", buildTimeSec, throughput, memoryMb);

    Map<String, String> params = new LinkedHashMap<>(buildParams);
    params.put("N", String.valueOf(n));
    params.put("dim", String.valueOf(dim));

    return BenchmarkResult.builder("synthetic-random", algorithm)
        .buildParams(params)
        .searchParams(Map.of())
        .buildTimeSeconds(buildTimeSec)
        .indexSizeMb(memoryMb)
        .extra(
            Map.of(
                "throughput_vec_per_sec",
                String.format("%.0f", throughput),
                "N",
                String.valueOf(n)))
        .build();
  }

  private static float[][] generateCorpus(int n, int dim) {
    Random rng = new Random(SEED);
    float[][] corpus = new float[n][dim];
    for (float[] row : corpus) {
      for (int d = 0; d < dim; d++) {
        row[d] = rng.nextFloat() * 2f - 1f;
      }
    }
    return corpus;
  }
}
