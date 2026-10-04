/* Copyright 2026 Integrallis Software, LLC. Licensed under the Apache License, Version 2.0. */
import com.integrallis.vectors.core.*;
import com.integrallis.vectors.db.*;
import com.integrallis.vectors.db.index.*;
import com.integrallis.vectors.db.storage.*;
import com.integrallis.vectors.hnsw.*;
import java.lang.foreign.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.function.LongSupplier;

/** Frozen graph comparisons: the same IDs, query budgets and float score bits in both arms. */
public class QueryPerformance {
  static volatile long sink;
  static float[][] read(Path path) throws Exception {
    var b = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
    float[][] rows = new float[b.getInt()][b.getInt()];
    for (var row : rows) for (int i = 0; i < row.length; i++) row[i] = b.getFloat();
    if (b.hasRemaining()) throw new AssertionError("trailing data");
    return rows;
  }
  static long hash(long h, int[] ids, float[] scores) {
    for (int i = 0; i < ids.length; i++) h = 31 * (31 * h + ids[i]) + Float.floatToRawIntBits(scores[i]);
    return h;
  }
  static void measure(String name, int queryCount, int passes, LongSupplier task) {
    measure(name, queryCount, passes, 12, task);
  }
  static void measure(String name, int queryCount, int passes, int warmPasses, LongSupplier task) {
    long start = System.nanoTime();
    int warm = 0;
    while (warm < warmPasses || System.nanoTime() - start < 3_000_000_000L) { sink = task.getAsLong(); warm++; }
    System.out.printf("WARMUP,%s,%d,%d%n", name, warm, System.nanoTime()-start);
    var memory = (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
    if (memory.isThreadAllocatedMemorySupported()) {
      memory.setThreadAllocatedMemoryEnabled(true);
      long tid = Thread.currentThread().threadId(), before = memory.getThreadAllocatedBytes(tid);
      sink = task.getAsLong();
      System.out.printf("ALLOCATION,%s,%d,%d%n", name, memory.getThreadAllocatedBytes(tid)-before, queryCount);
    }
    long expected = task.getAsLong();
    for (int sample = 0; sample < 7; sample++) {
      start = System.nanoTime();
      for (int p = 0; p < passes; p++) {
        long got = task.getAsLong();
        if (got != expected) throw new AssertionError("unstable result " + name);
        sink = got;
      }
      System.out.printf("RESULT,%s,%d,%d,%d,%d%n", name, sample, System.nanoTime()-start, expected, queryCount*passes);
    }
  }
  public static void main(String[] args) throws Exception {
    String name = args[0];
    Path input = Path.of(args[1]);
    float[][] data, queries;
    var metric = SimilarityFunction.valueOf(args[4]);
    if (Files.isDirectory(input)) {
      var m = Manifest.readFrom(input.resolve("manifest.bin"));
      data = new float[Math.toIntExact(m.liveCount())][m.dimension()];
      try (var arena = Arena.ofConfined()) {
        var s = MemorySegmentVectors.open(input.resolve("vectors.bin"), data.length, m.dimension(), arena);
        for (int i = 0; i < data.length; i++) MemorySegment.copy(s.vectorSlice(i), ValueLayout.JAVA_FLOAT, 0, data[i], 0, m.dimension());
      }
      if (Files.isRegularFile(Path.of(args[2]))) {
        queries = read(Path.of(args[2]));
      } else {
        queries = new float[256][];
        for (int q = 0; q < queries.length; q++) {
          queries[q] = data[(int)((long) q * data.length / queries.length)].clone();
          queries[q][q % queries[q].length] += .001f;
        }
      }
    } else {
      data = read(input); queries = read(Path.of(args[2]));
    }
    int dim = data[0].length;
    String mode = args[5];
    if (mode.startsWith("flat")) {
      try (var c = VectorCollection.builder().dimension(dim).metric(metric).indexType(IndexType.FLAT).autoCommitThreshold(Integer.MAX_VALUE).build()) {
        for (int i = 0; i < data.length; i++) c.add(Document.of("row-"+i, data[i]));
        c.commit(); System.gc();
        var requests = new ArrayList<SearchRequest>();
        for (int q = 0; q < 32; q++) requests.add(SearchRequest.builder(queries[q], 10).build());
        measure(name+"-flat-public-single", 32, 1, () -> {
          long h=1;
          for (var r : requests) for (var hit : c.search(r).hits()) h=31*(31*h+hit.id().hashCode())+Float.floatToRawIntBits(hit.score());
          return h;
        });
        if (!mode.equals("flat-single")) for (int count : new int[]{4,16,32}) {
          var batchRequests = requests.subList(0,count);
          measure(name+"-flat-public-batch"+count, count, 1, () -> {
            long h=1;
            for (var r : c.searchBatch(batchRequests)) for (var hit : r.hits()) h=31*(31*h+hit.id().hashCode())+Float.floatToRawIntBits(hit.score());
            return h;
          });
        }
        if (!mode.equals("flat-single")) for (int count : new int[]{4,16}) {
          try (var executor = java.util.concurrent.Executors.newFixedThreadPool(count)) {
            measure(name+"-flat-independent"+count, count, 1, () -> {
              var start = new java.util.concurrent.CountDownLatch(1);
              var jobs = new ArrayList<java.util.concurrent.Future<com.integrallis.vectors.db.SearchResult>>();
              for (var request : requests.subList(0,count))
                jobs.add(executor.submit(() -> { start.await(); return c.search(request); }));
              start.countDown();
              long h=1;
              try {
                for (var job : jobs) for (var hit : job.get().hits())
                  h=31*(31*h+hit.id().hashCode())+Float.floatToRawIntBits(hit.score());
              } catch (Exception e) { throw new RuntimeException(e); }
              return h;
            });
          }
        }
      }
      return;
    }
    if (mode.equals("public")) {
      try (var c = VectorCollection.builder().dimension(dim).metric(metric).indexType(IndexType.HNSW).storagePath(input.getParent()).build()) {
        System.gc();
        for (int ef : new int[]{32,100,128,512}) {
          var requests = new ArrayList<SearchRequest>();
          for (var q : queries) requests.add(SearchRequest.builder(q,10).searchListSize(ef).build());
          measure(name+"-hnsw-public-ef"+ef, requests.size(), 4, () -> {
            long h=1;
            for (var request : requests) for (var hit : c.search(request).hits())
              h=31*(31*h+hit.id().hashCode())+Float.floatToRawIntBits(hit.score());
            return h;
          });
        }
      }
      return;
    }
    var graph = HnswGraphCodec.decode(Files.readAllBytes(Path.of(args[3])));
    float[][] graphRows = Arrays.copyOf(data, graph.size());
    RandomAccessVectors source = new InMemoryVectors(graphRows);
    try (var arena = Arena.ofConfined()) {
      if (mode.startsWith("mapped")) {
        var segment = arena.allocate((long) graphRows.length * dim * 4);
        for (int i=0;i<graphRows.length;i++) MemorySegment.copy(graphRows[i],0,segment,ValueLayout.JAVA_FLOAT,(long)i*dim*4,dim);
        source = new RandomAccessVectors() {
          public int size(){return graphRows.length;}
          public int dimension(){return dim;}
          public float[] getVector(int i){return graphRows[i];}
          public boolean sharesReturnBuffer(){return false;}
          public boolean supportsSegments(){return true;}
          public MemorySegment vectorSegment(int i){return segment.asSlice((long)i*dim*4,(long)dim*4);}
          public MemorySegment vectorSegmentStorage(){return segment;}
          public long vectorSegmentOffset(int i){return (long)i*dim*4;}
        };
      }
      var index = HnswIndex.ofPrebuilt(graph, source, metric);
      System.gc();
      if (!mode.endsWith("-filtered")) for (int ef : new int[]{32,100,128,512}) {
        measure(name+"-hnsw-"+mode+"-ef"+ef, queries.length, 4, () -> {
          long h=1;
          for (var q : queries) { var r=index.search(q,10,ef); h=hash(h,r.nodeIds(),r.scores()); }
          return h;
        });
      }
      // The diagnostic mode warms at least 32k query/searcher constructions before timing.
      // Retain the original fresh-searcher workload and measure reuse separately.
      int filteredWarmPasses = mode.endsWith("-filtered") ? 128 : 12;
      measure(name+"-hnsw-"+mode+"-filtered128", queries.length, 4, filteredWarmPasses, () -> {
        long h=1;
        for (var q : queries) { var r=index.searcher().searchFiltered(q,10,128,id -> id%5==0); h=hash(h,r.nodeIds(),r.scores()); }
        return h;
      });
      if (mode.endsWith("-filtered")) {
        var searcher = index.searcher();
        measure(name+"-hnsw-"+mode+"-reused128", queries.length, 4, 128, () -> {
          long h=1;
          for (var q : queries) { var r=searcher.searchFiltered(q,10,128,id -> id%5==0); h=hash(h,r.nodeIds(),r.scores()); }
          return h;
        });
      }
    }
  }
}
