import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.db.index.*;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import com.integrallis.vectors.hnsw.*;
import com.integrallis.vectors.storage.store.VectorStoreWriter;
import java.lang.foreign.Arena;
import java.lang.management.ManagementFactory;
import java.nio.file.*;
import java.util.*;

/** Identical public-API workload compiled once and run against both revisions. */
public final class SearchPerformance {
  static volatile long sink;

  static float[][] rows(int n, int dim, long seed) {
    var random = new SplittableRandom(seed);
    var rows = new float[n][dim];
    for (var row : rows) for (int d = 0; d < dim; d++) row[d] = (float) random.nextDouble(-1, 1);
    return rows;
  }

  static long digest(int[] ids, float[] scores) {
    long hash = 1;
    for (int i = 0; i < ids.length; i++)
      hash = 31 * (31 * hash + ids[i]) + Float.floatToIntBits(scores[i]);
    return hash;
  }

  static long flatPass(IndexSpi index, float[][] queries, boolean batch) {
    long h = 1;
    if (batch) {
      for (var result : index.searchBatch(queries, 10, 128, 1f))
        h = 31 * h + digest(result.ordinals(), result.scores());
    } else
      for (var query : queries) {
        var result = index.search(query, 10, 128, 1f);
        h = 31 * h + digest(result.ordinals(), result.scores());
      }
    sink = h;
    return h;
  }

  static void flat(String kind, IndexSpi index, float[][] queries, int dim, boolean batch) {
    for (int i = 0; i < 8; i++) flatPass(index, queries, batch);
    long hash = flatPass(index, queries, batch);
    for (int round = 0; round < 7; round++) {
      long start = System.nanoTime();
      long got = flatPass(index, queries, batch);
      long elapsed = System.nanoTime() - start;
      if (got != hash) throw new AssertionError("unstable flat results");
      System.out.printf(
          Locale.ROOT,
          "flat-%s%s,%d,10000,10,%d,%d,%d%n",
          kind,
          batch ? "-batch" : "",
          dim,
          round,
          elapsed,
          hash);
    }
  }

  static long graphPass(HnswIndex index, float[][] queries, int ef, boolean filtered) {
    long h = 1;
    for (var query : queries) {
      var result =
          filtered
              ? index.searchFiltered(query, 10, ef, i -> (i & 1) == 0)
              : index.search(query, 10, ef);
      h = 31 * h + digest(result.nodeIds(), result.scores());
    }
    sink = h;
    return h;
  }

  public static void main(String[] args) throws Exception {
    Path output = Path.of(args[0]);
    Files.createDirectories(output);
    System.out.println("workload,dimensions,rows,k,round,nanos,result_digest");
    for (int dim : new int[] {128, 512, 768}) {
      var data = rows(10000, dim, 92641);
      var queries = rows(64, dim, 638221);
      var heap = new FlatScanAdapter();
      heap.build(data, SimilarityFunction.COSINE);
      flat("heap", heap, queries, dim, false);
      flat("heap", heap, queries, dim, true);
      Path file = output.resolve("vectors-" + dim + ".bin");
      try (var writer = VectorStoreWriter.open(file, dim, VectorEncoding.FLOAT32)) {
        for (var row : data) writer.writeVector(row);
      }
      try (var arena = Arena.ofConfined()) {
        var store = MemorySegmentVectors.open(file, data.length, dim, arena);
        flat(
            "mapped",
            new MappedFlatScanAdapter(store, SimilarityFunction.COSINE),
            queries,
            dim,
            false);
      }
      Files.delete(file);
    }
    var data = rows(20000, 128, 72931);
    var queries = rows(1000, 128, 128371);
    for (var matrix : new float[][][] {data, queries})
      for (var row : matrix) {
        double norm = 0;
        for (float value : row) norm += (double) value * value;
        float scale = (float) (1 / Math.sqrt(norm));
        for (int d = 0; d < row.length; d++) row[d] *= scale;
      }
    var source = new InMemoryVectors(data);
    Path graphFile = output.getParent().resolve("fixed-graph.bin");
    if (!Files.exists(graphFile)) {
      var built =
          HnswGraphBuilder.create(16, 200, source, SimilarityFunction.DOT_PRODUCT, 421).build();
      Files.write(graphFile, com.integrallis.vectors.db.storage.HnswGraphCodec.encode(built));
    }
    var graph =
        com.integrallis.vectors.db.storage.HnswGraphCodec.decode(Files.readAllBytes(graphFile));
    var index = HnswIndex.ofPrebuilt(graph, source, SimilarityFunction.DOT_PRODUCT);
    var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    bean.setThreadAllocatedMemoryEnabled(true);
    for (int ef : new int[] {32, 128, 512})
      for (boolean filtered : new boolean[] {false, true}) {
        for (int warm = 0; warm < 3; warm++) graphPass(index, queries, ef, filtered);
        long hash = graphPass(index, queries, ef, filtered);
        for (int round = 0; round < 7; round++) {
          long start = System.nanoTime();
          long got = graphPass(index, queries, ef, filtered);
          long elapsed = System.nanoTime() - start;
          if (got != hash) throw new AssertionError("unstable graph results");
          System.out.printf(
              Locale.ROOT,
              "hnsw-%d%s,128,20000,10,%d,%d,%d%n",
              ef,
              filtered ? "-filtered" : "",
              round,
              elapsed,
              hash);
        }
      }
    long before = bean.getThreadAllocatedBytes(Thread.currentThread().threadId());
    for (int i = 0; i < 20; i++) sink = System.identityHashCode(index.searcher());
    long allocated = bean.getThreadAllocatedBytes(Thread.currentThread().threadId()) - before;
    System.out.println("SEARCHER_ALLOCATION_BYTES=" + allocated / 20);
  }
}
