/* Copyright 2026 Integrallis Software, LLC. Licensed under the Apache License, Version 2.0. */
import com.integrallis.vectors.core.*;
import com.integrallis.vectors.db.*;
import com.integrallis.vectors.db.storage.*;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.*;

/** Exercises VectorCollection's public API, including dispatch, projection and virtual workers. */
public class CollectionBatchPerformance {
  static volatile long sink;

  static long pass(VectorCollection collection, List<SearchRequest> requests, boolean batch) {
    var results =
        batch
            ? collection.searchBatch(requests)
            : requests.stream().map(collection::search).toList();
    long hash = 1;
    for (var result : results)
      for (var hit : result.hits())
        hash = 31 * (31 * hash + hit.id().hashCode()) + Float.floatToIntBits(hit.score());
    return sink = hash;
  }

  public static void main(String[] args) throws Exception {
    String name = args[0];
    Path input = Path.of(args[1]);
    float[][] data, queries;
    if (Files.isDirectory(input)) {
      var m = Manifest.readFrom(input.resolve("manifest.bin"));
      int n = Math.toIntExact(m.liveCount()), d = m.dimension();
      data = new float[n][d];
      try (var arena = Arena.ofConfined()) {
        var store = MemorySegmentVectors.open(input.resolve("vectors.bin"), n, d, arena);
        for (int i = 0; i < n; i++)
          MemorySegment.copy(store.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0, data[i], 0, d);
      }
      queries = new float[64][d];
      for (int q = 0; q < queries.length; q++) {
        queries[q] = data[(int) ((long) q * n / queries.length)].clone();
        queries[q][q % d] += .001f;
      }
    } else {
      data = FlatBatchPerformance.read(input);
      queries = FlatBatchPerformance.read(Path.of(args[2]));
    }
    try (var collection =
        VectorCollection.builder()
            .dimension(data[0].length)
            .metric(SimilarityFunction.valueOf(args[3]))
            .indexType(IndexType.FLAT)
            .autoCommitThreshold(Integer.MAX_VALUE)
            .build()) {
      for (int i = 0; i < data.length; i++) collection.add(Document.of("row-" + i, data[i]));
      collection.commit();
      for (String count : args[4].split(",")) {
        int nq = Integer.parseInt(count);
        if (nq > queries.length) throw new IllegalArgumentException("not enough queries");
        var requests = new ArrayList<SearchRequest>();
        for (int q = 0; q < nq; q++) requests.add(SearchRequest.builder(queries[q], 10).build());
        for (boolean batch : new boolean[] {false, true}) {
          for (int warm = 0; warm < Integer.parseInt(args[5]); warm++)
            pass(collection, requests, batch);
          long hash = pass(collection, requests, batch);
          for (int sample = 0; sample < Integer.parseInt(args[6]); sample++) {
            long start = System.nanoTime(),
                got = pass(collection, requests, batch),
                ns = System.nanoTime() - start;
            if (got != hash) throw new AssertionError("unstable output");
            System.out.printf(
                Locale.ROOT,
                "FLAT,%s,%s,%d,%d,%d,%d,%d,%d%n",
                name,
                batch ? "batch" : "single",
                data.length,
                data[0].length,
                nq,
                sample,
                ns,
                hash);
          }
        }
      }
    }
  }
}
