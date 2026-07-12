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
package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
  // Zero-copy segment scoring: when the store exposes stable mmap slices (MappedBuildVectors), score
  // directly off the slice instead of allocating a fresh float[dim] per candidate. This is what makes
  // an mmap-backed build viable — the naive getVector() path allocates ~10^8 float[] and is GC-bound.
  private final boolean useSegments;
  private final int dimension;

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
    this.useSegments = vectors.supportsSegments();
    this.dimension = vectors.dimension();
  }

  /**
   * Scores {@code query} against the vector at {@code nodeId}. On the zero-copy segment path this
   * reads the stored vector's mmap slice directly (no per-candidate {@code float[]} allocation);
   * otherwise it falls back to the heap-array path. The query segment is uploaded once per insert
   * into {@code ctx.queryScratchSeg}.
   */
  private float scoreNode(float[] query, WorkContext ctx, int nodeId) {
    return useSegments
        ? similarityFunction.compare(ctx.queryScratchSeg, vectors.vectorSegment(nodeId), dimension)
        : similarityFunction.compare(query, vectors.getVector(nodeId));
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
    var threadCtx =
        ThreadLocal.withInitial(
            () -> new WorkContext(n, efConstruction, maxNbrs, dimension, useSegments));

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
    // Zero-copy segment-scoring scratch (null unless the store supports segments). queryScratchSeg is
    // an off-heap copy of the current insert's query (refilled once per insert, not per candidate);
    // rowSegs holds reusable zero-copy vectorSegment() slices for the fused segment GEMV.
    final MemorySegment queryScratchSeg;
    final MemorySegment[] rowSegs;

    WorkContext(int maxNodes, int ef, int maxNeighbors, int dimension, boolean useSegments) {
      visited = new BitSet(maxNodes);
      candidates = new NodeQueue(ef * 2, false);
      results = new NodeQueue(ef * 2, true);
      tmpIds = new int[maxNeighbors];
      pool = new float[maxNeighbors][];
      kernelOut = new float[maxNeighbors];
      batchIds = new int[maxNeighbors];
      batchScores = new float[maxNeighbors];
      if (useSegments) {
        // Arena.ofAuto(): GC-managed, lives as long as this per-thread WorkContext. Allocated ONCE.
        this.queryScratchSeg = Arena.ofAuto().allocate((long) dimension * Float.BYTES);
        this.rowSegs = new MemorySegment[maxNeighbors];
      } else {
        this.queryScratchSeg = null;
        this.rowSegs = null;
      }
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
    if (useSegments) {
      // Upload the query into the off-heap scratch ONCE per insert; every candidate score below
      // reads it against a zero-copy mmap slice, so no float[] is allocated per candidate.
      MemorySegment.copy(query, 0, ctx.queryScratchSeg, ValueLayout.JAVA_FLOAT, 0L, dimension);
    }

    // Phase 1: greedy descent from epLevel down to level+1
    int currentBest = ep;
    for (int layer = epLevel; layer > level; layer--) {
      currentBest = greedyConcurrent(query, currentBest, layer, graph, locks, ctx, nodeId);
    }

    // Phase 2: beam search + backlinks at each insertion layer
    int insertTopLayer = Math.min(level, epLevel);
    int[] entryPoints = {currentBest};

    for (int layer = insertTopLayer; layer >= 0; layer--) {
      int maxConn = (layer == 0) ? graph.maxConnections0() : maxConnections;

      NeighborArray searchResults =
          searchLayerConcurrent(
              query, entryPoints, efConstruction, layer, graph, locks, ctx, nodeId);

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
        // Reuse the forward-edge score: the metric is symmetric, so compare(nbr, query) is
        // bit-identical to the score already carried for this neighbor (compare(query, nbr)).
        float score = neighbors.score(i);
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

  /**
   * Single-best greedy walk at the given layer (used for upper-layer descent). {@code self} is the
   * node being inserted; it is never a valid move target — because all neighbor arrays exist from
   * the start, a concurrent insert may have already back-linked {@code self} into a node we visit,
   * and {@code self} scores maximal self-similarity, which would otherwise pull the walk onto
   * itself and ultimately produce a self-loop edge.
   */
  private int greedyConcurrent(
      float[] query,
      int entry,
      int layer,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx,
      int self) {
    int current = entry;
    float currentScore = scoreNode(query, ctx, current);
    boolean improved = true;
    while (improved) {
      improved = false;
      int nCount = snapshotNeighbors(current, layer, graph, locks, ctx.tmpIds);
      for (int i = 0; i < nCount; i++) {
        int nbr = ctx.tmpIds[i];
        if (nbr == self) continue;
        float s = scoreNode(query, ctx, nbr);
        if (s > currentScore) {
          currentScore = s;
          current = nbr;
          improved = true;
        }
      }
    }
    return current;
  }

  /**
   * ef-limited beam search; results returned as a descending-score NeighborArray. {@code self} (the
   * node being inserted) is pre-marked visited so it can never enter the candidate/result set: a
   * concurrent insert may have already back-linked {@code self} into the graph, and admitting it
   * here would select the node as its own neighbour (a self-loop). See {@link #greedyConcurrent}.
   */
  private NeighborArray searchLayerConcurrent(
      float[] query,
      int[] entryPoints,
      int ef,
      int layer,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx,
      int self) {

    ctx.visited.clear();
    ctx.candidates.clear();
    ctx.results.clear();
    ctx.visited.set(self); // never admit the node being inserted into its own neighbour set

    for (int ep : entryPoints) {
      if (!ctx.visited.get(ep)) {
        ctx.visited.set(ep);
        float score = scoreNode(query, ctx, ep);
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
        if (useSegments) ctx.rowSegs[batch] = vectors.vectorSegment(nbr);
        else if (useBulk) ctx.pool[batch] = vectors.getVector(nbr);
        batch++;
      }
      if (batch == 0) continue;
      if (useSegments) {
        // Zero-copy fused GEMV: score all gathered mmap slices against the query in one SIMD pass.
        FusedSimilarity.bulkCompareSegments(
            similarityFunction, query, ctx.rowSegs, dimension, ctx.kernelOut, ctx.batchScores, batch);
      } else if (useBulk) {
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
        // Single sift-down eviction; only explore the neighbor if it entered the result beam.
        if (ctx.results.insertWithOverflow(nbr, score, ef)) {
          ctx.candidates.add(nbr, score);
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
        Throwable cause = e.getCause();
        throw new RuntimeException(cause != null ? cause : e);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException(e);
      }
    }
  }
}
