package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Multi-threaded HNSW graph builder using per-node {@link ReentrantLock}s and thread-local scratch
 * buffers to parallelize node insertion.
 *
 * <p>Build phases:
 *
 * <ol>
 *   <li><b>Pre-phase (sequential)</b>: Assign random levels for all nodes; allocate all neighbor
 *       arrays; select the highest-level node as the global entry point. Because levels are
 *       pre-assigned in the same sequential order as {@link HnswGraphBuilder}, the final entry
 *       point matches what the sequential builder would produce.
 *   <li><b>Concurrent phase</b>: Insert each non-entry node in parallel. Neighbor-list reads are
 *       snapshotted under the owner's lock to prevent reading a partially-written array; backlink
 *       writes acquire the target node's lock for the duration of the insert + prune.
 * </ol>
 *
 * <p>Instances are single-use: call {@link #build()} or {@link #build(int)} exactly once.
 */
public final class ConcurrentHnswGraphBuilder {

  private final int maxConnections;
  private final int efConstruction;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  private final RandomLevelGenerator levelGenerator;

  // Fused bulk scoring is safe only when the backing store returns stable vector references.
  private final boolean useBulk;

  private ConcurrentHnswGraphBuilder(
      int maxConnections,
      int efConstruction,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      long seed) {
    this.maxConnections = maxConnections;
    this.efConstruction = efConstruction;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.levelGenerator = new RandomLevelGenerator(maxConnections, seed);
    this.useBulk = !vectors.sharesReturnBuffer();
  }

  /**
   * Creates a new builder.
   *
   * @param maxConnections M — max connections per upper layer (layer-0 uses 2*M)
   * @param efConstruction beam width during construction
   * @param vectors dataset
   * @param sim similarity function
   * @param seed random seed for level generation (deterministic)
   */
  public static ConcurrentHnswGraphBuilder create(
      int maxConnections,
      int efConstruction,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      long seed) {
    if (maxConnections <= 0) throw new IllegalArgumentException("maxConnections must be > 0");
    if (efConstruction <= 0) throw new IllegalArgumentException("efConstruction must be > 0");
    if (vectors == null) throw new NullPointerException("vectors must not be null");
    if (sim == null) throw new NullPointerException("sim must not be null");
    return new ConcurrentHnswGraphBuilder(maxConnections, efConstruction, vectors, sim, seed);
  }

  /** Builds using {@code Runtime.availableProcessors()} threads. */
  public HnswGraph build() {
    return build(Runtime.getRuntime().availableProcessors());
  }

  /**
   * Builds the HNSW graph using {@code parallelism} threads.
   *
   * @param parallelism number of worker threads
   * @return the completed graph
   */
  public HnswGraph build(int parallelism) {
    int n = vectors.size();

    // --- Phase 1: assign levels (same order as sequential builder) ---
    int[] levels = new int[n];
    int entryNode = 0;
    int maxLevel = 0;
    for (int i = 0; i < n; i++) {
      levels[i] = levelGenerator.nextLevel();
      if (levels[i] > maxLevel) {
        maxLevel = levels[i];
        entryNode = i;
      }
    }

    // --- Phase 1b: initialize all nodes and locks sequentially ---
    var graph = new HnswGraph(n, maxConnections);
    var locks = new ReentrantLock[n];
    for (int i = 0; i < n; i++) {
      graph.initNode(i, levels[i]);
      locks[i] = new ReentrantLock();
    }
    graph.setEntryNode(entryNode, maxLevel);

    if (n == 1) return graph;

    // --- Phase 2: concurrent insertion ---
    final int finalEntry = entryNode;
    final int finalMaxLevel = maxLevel;
    int maxNbrs = graph.maxConnections0() + 1;
    var threadCtx = ThreadLocal.withInitial(() -> new WorkContext(n, efConstruction, maxNbrs));

    try (ExecutorService exec = Executors.newFixedThreadPool(parallelism)) {
      List<Future<?>> futures = new ArrayList<>(n);
      for (int i = 0; i < n; i++) {
        if (i == finalEntry) continue;
        final int nodeId = i;
        final int level = levels[i];
        futures.add(
            exec.submit(
                () ->
                    insertConcurrent(
                        nodeId, level, finalEntry, finalMaxLevel, graph, locks, threadCtx.get())));
      }
      awaitAll(futures);
    }
    return graph;
  }

  // ---------------------------------------------------------------------------
  // Per-thread scratch buffers
  // ---------------------------------------------------------------------------

  private static final class WorkContext {
    final BitSet visited;
    final NodeQueue candidates; // max-heap
    final NodeQueue results; // min-heap
    final int[] tmpIds; // snapshot buffer for neighbor reads under lock
    // Fused-GEMV scratch: aliased row pool + kernel output + per-batch ids/scores.
    final float[][] pool;
    final float[] kernelOut;
    final int[] batchIds;
    final float[] batchScores;

    WorkContext(int maxNodes, int ef, int maxNeighbors) {
      visited = new BitSet(maxNodes);
      candidates = new NodeQueue(ef * 2, false);
      results = new NodeQueue(ef * 2, true);
      tmpIds = new int[maxNeighbors];
      pool = new float[maxNeighbors][];
      kernelOut = new float[maxNeighbors];
      batchIds = new int[maxNeighbors];
      batchScores = new float[maxNeighbors];
    }
  }

  // ---------------------------------------------------------------------------
  // Concurrent node insertion
  // ---------------------------------------------------------------------------

  private void insertConcurrent(
      int nodeId,
      int level,
      int ep,
      int epLevel,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx) {

    float[] query = vectors.getVector(nodeId);

    // Phase 1: greedy descent from epLevel down to level+1
    int currentBest = ep;
    for (int layer = epLevel; layer > level; layer--) {
      currentBest = greedyConcurrent(query, currentBest, layer, graph, locks, ctx);
    }

    // Phase 2: beam search + backlinks at each insertion layer
    int insertTopLayer = Math.min(level, epLevel);
    int[] entryPoints = {currentBest};

    for (int layer = insertTopLayer; layer >= 0; layer--) {
      int maxConn = (layer == 0) ? graph.maxConnections0() : maxConnections;

      NeighborArray searchResults =
          searchLayerConcurrent(query, entryPoints, efConstruction, layer, graph, locks, ctx);

      NeighborArray neighbors =
          NeighborSelector.selectDiverse(searchResults, maxConn, vectors, similarityFunction);

      // Forward edges: nodeId → selected neighbors.
      // Use insert() (not copyFrom) so any backlinks already added by concurrent threads are
      // preserved. Prune after inserting all forward neighbors to maintain the maxConn invariant.
      locks[nodeId].lock();
      try {
        NeighborArray nodeList = graph.getNeighbors(nodeId, layer);
        for (int i = 0; i < neighbors.size(); i++) {
          nodeList.insert(neighbors.node(i), neighbors.score(i));
        }
        if (nodeList.size() > maxConn) {
          NeighborArray pruned =
              NeighborSelector.selectDiverse(nodeList, maxConn, vectors, similarityFunction);
          nodeList.copyFrom(pruned);
        }
      } finally {
        locks[nodeId].unlock();
      }

      // Reverse edges: each neighbor ← nodeId (requires per-neighbor lock)
      for (int i = 0; i < neighbors.size(); i++) {
        int nbr = neighbors.node(i);
        float score = similarityFunction.compare(vectors.getVector(nbr), query);
        locks[nbr].lock();
        try {
          NeighborArray nList = graph.getNeighbors(nbr, layer);
          if (nList != null) {
            nList.insert(nodeId, score);
            if (nList.size() > maxConn) {
              NeighborArray pruned =
                  NeighborSelector.selectDiverse(nList, maxConn, vectors, similarityFunction);
              nList.copyFrom(pruned);
            }
          }
        } finally {
          locks[nbr].unlock();
        }
      }

      entryPoints = topNodes(searchResults, Math.min(efConstruction, searchResults.size()));
    }
  }

  // ---------------------------------------------------------------------------
  // Concurrent greedy + beam search
  // ---------------------------------------------------------------------------

  /** Single-best greedy walk at the given layer (used for upper-layer descent). */
  private int greedyConcurrent(
      float[] query,
      int entry,
      int layer,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx) {
    int current = entry;
    float currentScore = similarityFunction.compare(query, vectors.getVector(current));
    boolean improved = true;
    while (improved) {
      improved = false;
      int nCount = snapshotNeighbors(current, layer, graph, locks, ctx.tmpIds);
      for (int i = 0; i < nCount; i++) {
        float s = similarityFunction.compare(query, vectors.getVector(ctx.tmpIds[i]));
        if (s > currentScore) {
          currentScore = s;
          current = ctx.tmpIds[i];
          improved = true;
        }
      }
    }
    return current;
  }

  /** ef-limited beam search; results returned as a descending-score NeighborArray. */
  private NeighborArray searchLayerConcurrent(
      float[] query,
      int[] entryPoints,
      int ef,
      int layer,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx) {

    ctx.visited.clear();
    ctx.candidates.clear();
    ctx.results.clear();

    for (int ep : entryPoints) {
      if (!ctx.visited.get(ep)) {
        ctx.visited.set(ep);
        float score = similarityFunction.compare(query, vectors.getVector(ep));
        ctx.candidates.add(ep, score);
        ctx.results.add(ep, score);
      }
    }

    while (!ctx.candidates.isEmpty()) {
      long top = ctx.candidates.poll();
      float candScore = NodeQueue.score(top);
      int candId = NodeQueue.nodeId(top);

      if (ctx.results.size() >= ef && candScore < NodeQueue.score(ctx.results.peek())) break;

      int nCount = snapshotNeighbors(candId, layer, graph, locks, ctx.tmpIds);
      // Gather unvisited neighbours into a batch, then fused-score them in one SIMD call.
      int batch = 0;
      for (int i = 0; i < nCount; i++) {
        int nbr = ctx.tmpIds[i];
        if (ctx.visited.get(nbr)) continue;
        ctx.visited.set(nbr);
        ctx.batchIds[batch] = nbr;
        if (useBulk) ctx.pool[batch] = vectors.getVector(nbr);
        batch++;
      }
      if (batch == 0) continue;
      if (useBulk) {
        FusedSimilarity.bulkCompare(
            similarityFunction, query, ctx.pool, ctx.kernelOut, ctx.batchScores, batch);
      } else {
        for (int i = 0; i < batch; i++)
          ctx.batchScores[i] =
              similarityFunction.compare(query, vectors.getVector(ctx.batchIds[i]));
      }
      for (int i = 0; i < batch; i++) {
        int nbr = ctx.batchIds[i];
        float score = ctx.batchScores[i];
        if (ctx.results.size() < ef) {
          ctx.candidates.add(nbr, score);
          ctx.results.add(nbr, score);
        } else if (score > NodeQueue.score(ctx.results.peek())) {
          ctx.candidates.add(nbr, score);
          ctx.results.poll();
          ctx.results.add(nbr, score);
        }
      }
    }

    // Drain min-heap to descending NeighborArray
    int sz = ctx.results.size();
    var arr = new NeighborArray(Math.max(1, sz));
    int[] tmpN = new int[sz];
    float[] tmpS = new float[sz];
    for (int i = sz - 1; i >= 0; i--) {
      long e = ctx.results.poll();
      tmpN[i] = NodeQueue.nodeId(e);
      tmpS[i] = NodeQueue.score(e);
    }
    for (int i = 0; i < sz; i++) arr.insert(tmpN[i], tmpS[i]);
    return arr;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /**
   * Acquires the node's lock, copies neighbor IDs into {@code dest}, releases the lock.
   *
   * @return the number of neighbors copied
   */
  private int snapshotNeighbors(
      int nodeId, int layer, HnswGraph graph, ReentrantLock[] locks, int[] dest) {
    locks[nodeId].lock();
    try {
      NeighborArray na = graph.getNeighbors(nodeId, layer);
      if (na == null) return 0;
      int n = na.size();
      for (int i = 0; i < n; i++) dest[i] = na.node(i);
      return n;
    } finally {
      locks[nodeId].unlock();
    }
  }

  private int[] topNodes(NeighborArray arr, int n) {
    int count = Math.min(n, arr.size());
    int[] result = new int[count];
    for (int i = 0; i < count; i++) result[i] = arr.node(i);
    return result;
  }

  private static void awaitAll(List<Future<?>> futures) {
    for (var f : futures) {
      try {
        f.get();
      } catch (ExecutionException e) {
        throw new RuntimeException(e.getCause());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException(e);
      }
    }
  }
}
