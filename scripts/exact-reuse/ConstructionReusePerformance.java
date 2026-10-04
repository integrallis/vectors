/* Copyright 2026 Integrallis Software, LLC. Licensed under the Apache License, Version 2.0. */
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.storage.HnswGraphCodec;
import com.integrallis.vectors.hnsw.*;
import java.nio.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Fixed M/efConstruction, published held-out queries, and baseline-owned exact oracle. */
public class ConstructionReusePerformance {
  static volatile long sink;

  static long search(HnswIndex index, float[][] queries, int ef) {
    long hash = 1;
    for (var q : queries) {
      var r = index.search(q, 10, ef);
      for (int i = 0; i < r.nodeIds().length; i++)
        hash = 31 * (31 * hash + r.nodeIds()[i]) + Float.floatToIntBits(r.scores()[i]);
    }
    return sink = hash;
  }

  public static void main(String[] args) throws Exception {
    // train, test, rows, threads, seed, metric, oracle, warmup rows, search samples, graph output
    var data =
        Arrays.copyOf(FlatBatchPerformance.read(Path.of(args[0])), Integer.parseInt(args[2]));
    var queries = FlatBatchPerformance.read(Path.of(args[1]));
    int threads = Integer.parseInt(args[3]);
    long seed = Long.parseLong(args[4]);
    var metric = SimilarityFunction.valueOf(args[5]);
    Path oracleFile = Path.of(args[6]);
    if (!Files.exists(oracleFile)) {
      var flat = new FlatScanAdapter();
      flat.build(data, metric);
      var buffer = ByteBuffer.allocate(queries.length * 10 * 4).order(ByteOrder.LITTLE_ENDIAN);
      // Warm up exact scoring before storing the oracle; oracle creation is never timed.
      for (int i = 0; i < 8; i++) flat.search(queries[0], 10, 128, 1f);
      for (var q : queries) for (int id : flat.search(q, 10, 128, 1f).ordinals()) buffer.putInt(id);
      Files.write(oracleFile, buffer.array());
    }
    var bytes = ByteBuffer.wrap(Files.readAllBytes(oracleFile)).order(ByteOrder.LITTLE_ENDIAN);
    var truth = new int[queries.length][10];
    for (var row : truth) for (int i = 0; i < 10; i++) row[i] = bytes.getInt();
    // Use one worker for deterministic compilation warmup in both arms.
    for (int warm = 0; warm < 2; warm++)
      ConcurrentHnswGraphBuilder.create(
              16,
              200,
              new InMemoryVectors(
                  Arrays.copyOf(data, Math.min(data.length, Integer.parseInt(args[7])))),
              metric,
              seed)
          .build(1);
    var source = new InMemoryVectors(data);
    long start = System.nanoTime();
    var graph = ConcurrentHnswGraphBuilder.create(16, 200, source, metric, seed).build(threads);
    long elapsed = System.nanoTime() - start;
    byte[] encodedGraph = HnswGraphCodec.encode(graph);
    String graphHash =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(encodedGraph));
    System.out.printf(
        Locale.ROOT,
        "BUILD,%d,%d,%d,%s,%d,%s%n",
        data.length,
        threads,
        seed,
        metric,
        elapsed,
        graphHash);
    var index = HnswIndex.ofPrebuilt(graph, source, metric);
    for (int ef : new int[] {32, 128, 512}) {
      for (int warm = 0; warm < 8; warm++) search(index, queries, ef);
      int hits = 0;
      for (int q = 0; q < queries.length; q++)
        for (int id : index.search(queries[q], 10, ef).nodeIds())
          for (int expected : truth[q])
            if (id == expected) {
              hits++;
              break;
            }
      System.out.printf(
          Locale.ROOT,
          "RECALL,%d,%d,%d,%.8f%n",
          ef,
          hits,
          queries.length * 10,
          hits / (queries.length * 10.0));
      long hash = search(index, queries, ef);
      for (int sample = 0; sample < Integer.parseInt(args[8]); sample++) {
        start = System.nanoTime();
        long got = search(index, queries, ef);
        elapsed = System.nanoTime() - start;
        if (got != hash) throw new AssertionError("unstable search results");
        System.out.printf(Locale.ROOT, "SEARCH,%d,%d,%d,%d%n", ef, sample, elapsed, hash);
      }
    }
    if (args.length > 9) Files.write(Path.of(args[9]), encodedGraph);
  }
}
