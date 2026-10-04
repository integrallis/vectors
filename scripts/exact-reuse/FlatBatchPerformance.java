/* Copyright 2026 Integrallis Software, LLC. Licensed under the Apache License, Version 2.0. */
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.FlatScanAdapter;
import com.integrallis.vectors.db.storage.*;
import java.lang.foreign.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Exact result digests plus flat batch timings. Corpus and shape are explicit CLI inputs. */
public class FlatBatchPerformance {
  static volatile long sink;

  static float[][] read(Path path) throws Exception {
    var b = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
    int n = b.getInt(), d = b.getInt();
    var rows = new float[n][d];
    for (var row : rows) for (int i = 0; i < d; i++) row[i] = b.getFloat();
    if (b.hasRemaining()) throw new IllegalArgumentException("trailing bytes");
    return rows;
  }

  static long pass(FlatScanAdapter index, float[][] queries, boolean batch) {
    long hash = 1;
    if (batch) {
      for (var r : index.searchBatch(queries, 10, 128, 1f))
        for (int i = 0; i < r.ordinals().length; i++)
          hash = 31 * (31 * hash + r.ordinals()[i]) + Float.floatToIntBits(r.scores()[i]);
    } else {
      for (var q : queries) {
        var r = index.search(q, 10, 128, 1f);
        for (int i = 0; i < r.ordinals().length; i++)
          hash = 31 * (31 * hash + r.ordinals()[i]) + Float.floatToIntBits(r.scores()[i]);
      }
    }
    return sink = hash;
  }

  public static void main(String[] args) throws Exception {
    // name, train fbin or generation directory, test fbin or '-', metric, query counts, warmups,
    // samples
    String name = args[0];
    Path input = Path.of(args[1]);
    float[][] data, allQueries;
    if (Files.isDirectory(input)) {
      var m = Manifest.readFrom(input.resolve("manifest.bin"));
      int n = Math.toIntExact(m.liveCount()), d = m.dimension();
      data = new float[n][d];
      try (var arena = Arena.ofConfined()) {
        var store = MemorySegmentVectors.open(input.resolve("vectors.bin"), n, d, arena);
        for (int i = 0; i < n; i++)
          MemorySegment.copy(store.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0, data[i], 0, d);
      }
      allQueries = new float[64][d];
      for (int q = 0; q < 64; q++) {
        allQueries[q] = data[(int) ((long) q * n / 64)].clone();
        allQueries[q][q % d] += .001f;
      }
    } else {
      data = read(input);
      allQueries = read(Path.of(args[2]));
    }
    var index = new FlatScanAdapter();
    index.build(data, SimilarityFunction.valueOf(args[3]));
    for (String count : args[4].split(",")) {
      int nq = Integer.parseInt(count);
      var queries = Arrays.copyOf(allQueries, nq);
      if (nq > allQueries.length) throw new IllegalArgumentException("not enough queries");
      for (boolean batch : new boolean[] {false, true}) {
        for (int warm = 0; warm < Integer.parseInt(args[5]); warm++) pass(index, queries, batch);
        long hash = pass(index, queries, batch);
        for (int sample = 0; sample < Integer.parseInt(args[6]); sample++) {
          long start = System.nanoTime(),
              got = pass(index, queries, batch),
              ns = System.nanoTime() - start;
          if (hash != got) throw new AssertionError("unstable output");
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
