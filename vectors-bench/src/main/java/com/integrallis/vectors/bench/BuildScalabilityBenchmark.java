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
import com.integrallis.vectors.bench.dataset.Hdf5Loader;
import com.integrallis.vectors.bench.report.BenchmarkReporter;
import com.integrallis.vectors.bench.report.BenchmarkResult;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

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
    int dim = args.length > 0 ? Integer.parseInt(args[0]) : Integer.getInteger("bench.dim", 128);

    int[] sizes = parseSizes(System.getProperty("bench.sizes"));
    Set<String> algos = parseAlgos(System.getProperty("bench.algo"));
    int hnswM = Integer.getInteger("bench.hnsw.m", 32);
    int hnswEf = Integer.getInteger("bench.hnsw.ef", 200);
    int vamanaR = Integer.getInteger("bench.vamana.r", 64);
    int vamanaL = Integer.getInteger("bench.vamana.l", 128);
    String datasetName = System.getProperty("bench.dataset"); // null => synthetic random

    List<BenchmarkResult> results = new ArrayList<>();
    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();

    // Preload HDF5 dataset once if requested; subsequent N values slice prefix rows.
    float[][] datasetCorpus = null;
    String corpusLabel = "synthetic-random";
    if (datasetName != null && !datasetName.isBlank()) {
      datasetCorpus = loadDatasetCorpus(datasetName);
      corpusLabel = datasetName;
      dim = datasetCorpus[0].length;
      System.out.printf(
          "Loaded dataset %s: %,d × %d from disk%n", datasetName, datasetCorpus.length, dim);
    }

    System.out.printf(
        "=== Build Scalability Benchmark (corpus=%s dim=%d sizes=%s algos=%s hnsw:M=%d,ef=%d) ===%n",
        corpusLabel, dim, Arrays.toString(sizes), algos, hnswM, hnswEf);

    for (int n : sizes) {
      System.out.printf("%n--- N = %,d ---%n", n);
      float[][] corpus;
      if (datasetCorpus != null) {
        if (n > datasetCorpus.length) {
          System.out.printf(
              "  skipping: N=%,d exceeds dataset size %,d%n", n, datasetCorpus.length);
          continue;
        }
        corpus = Arrays.copyOf(datasetCorpus, n); // prefix slice; shares row arrays
      } else {
        corpus = generateCorpus(n, dim);
      }

      if (algos.contains("hnsw")) {
        results.add(
            buildAndMeasure(
                corpusLabel,
                "hnsw",
                IndexType.HNSW,
                corpus,
                dim,
                n,
                Map.of("M", String.valueOf(hnswM), "efConstruction", String.valueOf(hnswEf)),
                memBean,
                hnswM,
                hnswEf,
                vamanaR,
                vamanaL));
      }

      if (algos.contains("vamana")) {
        results.add(
            buildAndMeasure(
                corpusLabel,
                "vamana",
                IndexType.VAMANA,
                corpus,
                dim,
                n,
                Map.of("R", String.valueOf(vamanaR), "L", String.valueOf(vamanaL), "alpha", "1.2"),
                memBean,
                hnswM,
                hnswEf,
                vamanaR,
                vamanaL));
      }

      if (algos.contains("flat")) {
        results.add(
            buildAndMeasure(
                corpusLabel,
                "flat",
                IndexType.FLAT,
                corpus,
                dim,
                n,
                Map.of(),
                memBean,
                hnswM,
                hnswEf,
                vamanaR,
                vamanaL));
      }

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
      String corpusLabel,
      String algorithm,
      IndexType indexType,
      float[][] corpus,
      int dim,
      int n,
      Map<String, String> buildParams,
      MemoryMXBean memBean,
      int hnswM,
      int hnswEf,
      int vamanaR,
      int vamanaL) {

    System.out.printf("  %s: building %,d vectors...", algorithm.toUpperCase(), n);

    System.gc();
    long heapBefore = memBean.getHeapMemoryUsage().getUsed();
    long gcCountBefore = totalGcCount();
    long gcTimeBeforeMs = totalGcTimeMs();

    long t0 = System.nanoTime();
    var builder =
        VectorCollection.builder()
            .dimension(dim)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(indexType);

    // Configure algorithm-specific params.
    if (indexType == IndexType.HNSW) {
      builder.hnswM(hnswM).hnswEfConstruction(hnswEf);
    } else if (indexType == IndexType.VAMANA) {
      builder.vamanaMaxDegree(vamanaR).vamanaSearchListSize(vamanaL).vamanaAlpha(1.2f);
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
    long gcCount = totalGcCount() - gcCountBefore;
    long gcTimeMs = totalGcTimeMs() - gcTimeBeforeMs;
    long heapAfter = memBean.getHeapMemoryUsage().getUsed();
    double memoryMb = Math.max(0, (heapAfter - heapBefore)) / (1024.0 * 1024.0);
    double throughput = n / buildTimeSec;
    double gcFraction = (gcTimeMs / 1000.0) / Math.max(buildTimeSec, 1e-9);

    col.close();

    System.out.printf(
        " %.1fs (%,.0f vec/s, %.0f MB, gc=%d@%.2fs=%.1f%%)%n",
        buildTimeSec, throughput, memoryMb, gcCount, gcTimeMs / 1000.0, gcFraction * 100);

    Map<String, String> params = new LinkedHashMap<>(buildParams);
    params.put("N", String.valueOf(n));
    params.put("dim", String.valueOf(dim));

    Map<String, String> extra = new LinkedHashMap<>();
    extra.put("throughput_vec_per_sec", String.format("%.0f", throughput));
    extra.put("N", String.valueOf(n));
    extra.put("gc_count", String.valueOf(gcCount));
    extra.put("gc_time_ms", String.valueOf(gcTimeMs));
    extra.put("gc_fraction_pct", String.format("%.2f", gcFraction * 100));

    return BenchmarkResult.builder(corpusLabel, algorithm)
        .buildParams(params)
        .searchParams(Map.of())
        .buildTimeSeconds(buildTimeSec)
        .indexSizeMb(memoryMb)
        .extra(extra)
        .build();
  }

  private static float[][] loadDatasetCorpus(String name) {
    try {
      Path hdf5 = DatasetRegistry.annBenchDataset(name);
      if (!java.nio.file.Files.exists(hdf5)) {
        if (DatasetDownloader.isKnownDataset(name)) {
          hdf5 = DatasetDownloader.ensureAvailable(name);
        } else {
          throw new IllegalArgumentException("Unknown dataset: " + name);
        }
      }
      return Hdf5Loader.readTrainVectors(hdf5);
    } catch (java.io.IOException e) {
      throw new RuntimeException("Failed to load dataset " + name, e);
    }
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

  private static int[] parseSizes(String spec) {
    if (spec == null || spec.isBlank()) {
      return new int[] {10_000, 50_000, 100_000, 500_000, 1_000_000};
    }
    String[] parts = spec.split(",");
    int[] sizes = new int[parts.length];
    for (int i = 0; i < parts.length; i++) {
      sizes[i] = Integer.parseInt(parts[i].trim().replace("_", ""));
    }
    return sizes;
  }

  private static Set<String> parseAlgos(String spec) {
    if (spec == null || spec.isBlank() || spec.equalsIgnoreCase("all")) {
      return Set.of("hnsw", "vamana", "flat");
    }
    Set<String> out = new java.util.LinkedHashSet<>();
    for (String s : spec.split(",")) {
      out.add(s.trim().toLowerCase());
    }
    return out;
  }

  private static long totalGcCount() {
    long total = 0;
    for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
      long c = gc.getCollectionCount();
      if (c > 0) total += c;
    }
    return total;
  }

  private static long totalGcTimeMs() {
    long total = 0;
    for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
      long t = gc.getCollectionTime();
      if (t > 0) total += t;
    }
    return total;
  }
}
