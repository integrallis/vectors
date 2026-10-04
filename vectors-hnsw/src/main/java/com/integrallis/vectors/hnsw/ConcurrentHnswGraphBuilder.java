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
  // Zero-copy segment scoring: when the store exposes stable mmap slices (MappedBuildVectors),
  // score
  // directly off the slice instead of allocating a fresh float[dim] per candidate. This is what
  // makes
  // an mmap-backed build viable — the naive getVector() path allocates ~10^8 float[] and is
  // GC-bound.
  private final boolean useSegments;
  private final int dimension;
  // Set for the duration of a segment-backed build; shared across worker threads. Null otherwise.
  private AsyncVectorPrefetcher prefetcher;

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
            () -> new WorkContext(n, efConstruction, maxNbrs, dimension, useSegments, !useBulk));

    // Async prefetch: when vectors are mmap-backed, page-in each popped candidate's neighbors on an
    // I/O pool so the NVMe reads overlap the SIMD scoring instead of faulting synchronously. This
    // is
    // what keeps the build from becoming NVMe-latency-bound once vectors.bin exceeds page cache
    // (307 GB at 100M). No-op-cheap when vectors are already resident.
    // Default OFF: a fault measurement (400K, ~25% page cache) showed async prefetch made the
    // memory-constrained mmap build ~15% SLOWER (638s vs 554s) — under a small cache the prefetched
    // pages evict before use and the I/O threads contend with the compute threads. Kept as an
    // opt-in
    // toggle for regimes it might help (fast NVMe + higher cache ratio), but not on by default.
    boolean prefetchEnabled =
        Boolean.parseBoolean(System.getProperty("vectors.hnsw.buildPrefetch", "false"));
    if (useSegments && prefetchEnabled) {
      prefetcher = new AsyncVectorPrefetcher(vectors, Math.max(2, parallelism));
    }
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
    } finally {
      if (prefetcher != null) {
        prefetcher.close();
        prefetcher = null;
      }
    }
    return graph;
  }

  /**
   * Extends an existing graph with the vectors appended after it, instead of rebuilding from
   * scratch.
   *
   * <p>Nodes {@code [0, firstNewOrdinal)} keep their levels and edge lists from {@code old} — the
   * ordinals are identical, so no remapping is needed. Nodes {@code [firstNewOrdinal, size)} are
   * assigned levels from the same exponential distribution a full build uses and inserted
   * concurrently through the ordinary Algorithm-1 path, including symmetric backlinks and diversity
   * pruning on their neighbours.
   *
   * <p>Cost is O(A · log N · M · d) for {@code A} appended vectors against a graph of {@code N},
   * where a full rebuild is O(N · log N · M · d). Committing an ingest in K batches therefore costs
   * one build rather than K builds of growing size.
   *
   * <p>Unlike {@link HnswGraphMerger#merge}, this does not remap ordinals and cannot remove nodes;
   * it is the append counterpart to that method's compaction.
   *
   * @param old the graph to extend; its {@code maxConnections} must match this builder's
   * @param firstNewOrdinal number of nodes carried over, which must equal {@code old.size()}
   * @param parallelism worker threads for inserting the appended nodes
   * @return a new graph holding the carried-over nodes plus the appended ones
   */
  public HnswGraph append(HnswGraph old, int firstNewOrdinal, int parallelism) {
    return append(old, firstNewOrdinal, parallelism, false);
  }

  /**
   * As {@link #append(HnswGraph, int, int)}, but skips recomputing the carried-over edge scores
   * when the caller knows {@code old} holds real similarities.
   *
   * <p>A graph decoded from {@code graph.bin} does not: {@code HnswGraphCodec} in {@code
   * vectors-db} stores node ids only and synthesises monotonically-decreasing scores on decode,
   * which is sound for search and wrong for insertion, since insertion and diversity pruning order
   * neighbours by score. Pass {@code true} only for a graph this process built and kept in memory.
   * Passing it wrongly degrades the graph silently.
   *
   * @param scoresAreReal whether {@code old}'s neighbour scores are real similarities
   */
  public HnswGraph append(
      HnswGraph old, int firstNewOrdinal, int parallelism, boolean scoresAreReal) {
    return append(old, firstNewOrdinal, parallelism, scoresAreReal, 0);
  }

  /**
   * As {@link #append(HnswGraph, int, int, boolean)}, allocating the successor with room to grow.
   *
   * <p>{@code capacityHint} is the number of nodes the result should be able to hold. Allocating
   * headroom lets later appends extend the same graph through {@link #appendInPlace} instead of
   * copying every carried node again, which is the difference between an ingest paying
   * O(collection) per commit and paying it once, amortised.
   */
  public HnswGraph append(
      HnswGraph old,
      int firstNewOrdinal,
      int parallelism,
      boolean scoresAreReal,
      int capacityHint) {
    if (old == null) {
      throw new NullPointerException("old must not be null");
    }
    int n = vectors.size();
    if (firstNewOrdinal < 0 || firstNewOrdinal > n) {
      throw new IllegalArgumentException(
          "firstNewOrdinal must be in [0, " + n + "]: " + firstNewOrdinal);
    }
    if (old.size() != firstNewOrdinal) {
      throw new IllegalArgumentException(
          "old graph holds " + old.size() + " nodes but firstNewOrdinal is " + firstNewOrdinal);
    }
    if (old.maxConnections() != maxConnections) {
      throw new IllegalArgumentException(
          "old graph was built with M="
              + old.maxConnections()
              + " but this builder uses M="
              + maxConnections);
    }
    if (n == 0) {
      return null;
    }
    if (firstNewOrdinal == 0) {
      return build(parallelism); // nothing to carry over
    }

    HnswGraph graph = new HnswGraph(Math.max(n, capacityHint), maxConnections);

    // --- Phase 1: carry the existing nodes over, ordinals unchanged ---
    int entryNode = old.entryNode();
    int maxLevel = old.maxLevel();
    for (int j = 0; j < firstNewOrdinal; j++) {
      graph.initNode(j, old.nodeLevel(j));
    }
    // Carried-over scores are recomputed unless the caller vouches for them. A graph decoded from
    // graph.bin carries *synthetic* monotonically-decreasing scores — HnswGraphCodec stores node
    // ids
    // only, which is sound while a committed graph is read-only but not once it is extended:
    // insertion and diversity pruning order neighbours by score, so carrying synthetic values
    // forward silently degrades the graph. Recomputing is O(N·M) distance computations, which is
    // why
    // a caller that kept its own freshly built graph passes scoresAreReal and skips it.
    float[] carriedScratch =
        !scoresAreReal && vectors.sharesReturnBuffer() ? new float[dimension] : null;
    for (int j = 0; j < firstNewOrdinal; j++) {
      int level = graph.nodeLevel(j);
      float[] self = scoresAreReal ? null : vectors.getVector(j);
      if (carriedScratch != null) {
        System.arraycopy(self, 0, carriedScratch, 0, dimension);
        self = carriedScratch;
      }
      for (int l = 0; l <= level; l++) {
        NeighborArray from = old.getNeighbors(j, l);
        if (from == null) {
          continue;
        }
        NeighborArray to = graph.getNeighbors(j, l);
        if (scoresAreReal) {
          // The source is already in descending-score order, so this is two array copies rather
          // than
          // one sorted insertion per neighbour. On a 100,000-node graph at M=16 that is the
          // difference between ~3.2 million insertions per commit and 100,000 copies.
          to.copyFrom(from);
          continue;
        }
        for (int i = 0; i < from.size(); i++) {
          int neighbour = from.node(i);
          to.insert(neighbour, similarityFunction.compare(self, vectors.getVector(neighbour)));
        }
      }
    }

    return insertAppended(graph, firstNewOrdinal, n, entryNode, maxLevel, parallelism, null, null);
  }

  /**
   * Extends a graph this builder's caller owns exclusively, in place.
   *
   * <p>No node is copied: the carried nodes are already in {@code graph}. The caller must guarantee
   * that nothing else can read the graph concurrently — a published generation's readers must have
   * their own copy or their own decode — because inserting an appended node also rewrites the
   * neighbour lists of existing nodes.
   *
   * @param graph a graph holding exactly {@code firstNewOrdinal} nodes with capacity for the
   *     builder's full vector count
   * @param firstNewOrdinal the first appended ordinal, which must equal {@code graph.size()}
   */
  public HnswGraph appendInPlace(HnswGraph graph, int firstNewOrdinal, int parallelism) {
    return appendInPlace(graph, firstNewOrdinal, parallelism, null, null);
  }

  /**
   * As {@link #appendInPlace(HnswGraph, int, int)}, reusing the caller's per-node lock table and
   * executor.
   *
   * <p>Both are per-collection, not per-commit. Insertion needs one lock per node and a pool of
   * workers; allocating them inside each append made a 40-commit ingest create two million locks
   * and forty thread pools, which is work proportional to the collection on every commit — the
   * shape this method exists to avoid.
   *
   * @param locks a table with at least {@code vectors.size()} entries, all non-null; null allocates
   *     one
   * @param executor workers for the insertion pass; null creates and closes a pool for this call
   */
  public HnswGraph appendInPlace(
      HnswGraph graph,
      int firstNewOrdinal,
      int parallelism,
      ReentrantLock[] locks,
      ExecutorService executor) {
    if (graph == null) {
      throw new NullPointerException("graph must not be null");
    }
    int n = vectors.size();
    if (graph.size() != firstNewOrdinal) {
      throw new IllegalArgumentException(
          "graph holds " + graph.size() + " nodes but firstNewOrdinal is " + firstNewOrdinal);
    }
    if (graph.capacity() < n) {
      throw new IllegalArgumentException(
          "graph capacity " + graph.capacity() + " cannot hold " + n + " nodes");
    }
    if (graph.maxConnections() != maxConnections) {
      throw new IllegalArgumentException(
          "graph was built with M="
              + graph.maxConnections()
              + " but this builder uses M="
              + maxConnections);
    }
    if (n == firstNewOrdinal) {
      return graph;
    }
    if (locks != null && locks.length < n) {
      throw new IllegalArgumentException(
          "lock table holds " + locks.length + " entries but " + n + " nodes are indexed");
    }
    return insertAppended(
        graph,
        firstNewOrdinal,
        n,
        graph.entryNode(),
        graph.maxLevel(),
        parallelism,
        locks,
        executor);
  }

  /** The insertion phase shared by {@link #append} and {@link #appendInPlace}. */
  private HnswGraph insertAppended(
      HnswGraph graph,
      int firstNewOrdinal,
      int n,
      int entryNode,
      int maxLevel,
      int parallelism,
      ReentrantLock[] providedLocks,
      ExecutorService providedExecutor) {

    int appended = n - firstNewOrdinal;
    int[] levels = new int[appended];
    int tallestNew = -1;
    int tallestNewLevel = -1;
    for (int i = 0; i < appended; i++) {
      levels[i] = levelGenerator.nextLevel();
      if (levels[i] > tallestNewLevel) {
        tallestNewLevel = levels[i];
        tallestNew = firstNewOrdinal + i;
      }
      graph.initNode(firstNewOrdinal + i, levels[i]);
    }
    // Searches performed *during* insertion must descend from a node that already has edges, so the
    // carried-over entry stays in place for the whole insertion pass. A taller appended node is
    // promoted afterwards. Skipping the insertion of a node because it is the entry — which is what
    // build() does, safely, on an empty graph — would leave the entry point with no edges at all
    // and
    // strand every search that starts there.
    graph.setEntryNode(entryNode, maxLevel);

    // --- Phase 3: insert the appended nodes through the ordinary path ---
    ReentrantLock[] locks = providedLocks;
    if (locks == null) {
      locks = new ReentrantLock[n];
      for (int j = 0; j < n; j++) {
        locks[j] = new ReentrantLock();
      }
    }
    final int finalEntry = entryNode;
    final int finalMaxLevel = maxLevel;
    int maxNbrs = graph.maxConnections0() + 1;
    var threadCtx =
        ThreadLocal.withInitial(
            () -> new WorkContext(n, efConstruction, maxNbrs, dimension, useSegments, !useBulk));

    ExecutorService exec =
        providedExecutor != null
            ? providedExecutor
            : Executors.newFixedThreadPool(Math.max(1, parallelism));
    try {
      List<Future<?>> futures = new ArrayList<>(appended);
      final ReentrantLock[] lockTable = locks;
      for (int i = 0; i < appended; i++) {
        final int nodeId = firstNewOrdinal + i;
        final int level = levels[i];
        futures.add(
            exec.submit(
                () ->
                    insertConcurrent(
                        nodeId,
                        level,
                        finalEntry,
                        finalMaxLevel,
                        graph,
                        lockTable,
                        threadCtx.get())));
      }
      awaitAll(futures);
    } finally {
      if (providedExecutor == null) {
        exec.close();
      }
    }

    // Promote a taller appended node to entry now that it is connected.
    if (tallestNewLevel > maxLevel) {
      graph.setEntryNode(tallestNew, tallestNewLevel);
    }

    // Reclaim the temporary overflow slot on every layer-0 array before the graph is serialised or
    // searched, so the M contract holds for carried-over nodes that gained backlinks above.
    int maxConn0 = graph.maxConnections0();
    for (int j = 0; j < n; j++) {
      NeighborArray na = graph.getNeighbors(j, 0);
      if (na != null) {
        na.trim(maxConn0);
      }
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
    // Zero-copy segment-scoring scratch (null unless the store supports segments). queryScratchSeg
    // is
    // an off-heap copy of the current insert's query (refilled once per insert, not per candidate);
    // rowSegs holds reusable zero-copy vectorSegment() slices for the fused segment GEMV.
    final float[] queryScratch;
    final MemorySegment queryScratchSeg;
    final MemorySegment[] rowSegs;

    WorkContext(
        int maxNodes,
        int ef,
        int maxNeighbors,
        int dimension,
        boolean useSegments,
        boolean sharesReturnBuffer) {
      queryScratch = !useSegments && sharesReturnBuffer ? new float[dimension] : null;
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

  private float[] insertionQuery(int nodeId, WorkContext ctx) {
    float[] query = vectors.getVector(nodeId);
    if (ctx.queryScratch != null && vectors.sharesReturnBuffer(nodeId)) {
      // One buffer per worker. Stable staged rows need no copy; mapped scratch rows do.
      System.arraycopy(query, 0, ctx.queryScratch, 0, dimension);
      return ctx.queryScratch;
    }
    return query;
  }

  private void insertConcurrent(
      int nodeId,
      int level,
      int ep,
      int epLevel,
      HnswGraph graph,
      ReentrantLock[] locks,
      WorkContext ctx) {

    float[] query = insertionQuery(nodeId, ctx);
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
      // Async-prefetch this candidate's neighbors so their mmap pages page-in on the I/O pool while
      // the gather + SIMD scoring below run — converting serial page faults into overlapped reads.
      AsyncVectorPrefetcher pf = prefetcher;
      if (pf != null) {
        // One task touches the whole neighbor list — batching keeps the pool submit rate low enough
        // that a 32-thread build doesn't serialise on the queue.
        pf.prefetch(ctx.tmpIds, nCount);
      }
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
            similarityFunction,
            query,
            ctx.rowSegs,
            dimension,
            ctx.kernelOut,
            ctx.batchScores,
            batch);
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

    return drainResults(ctx.results);
  }

  /** Converts the distinct result heap into the pruning candidate order. */
  static NeighborArray drainResults(NodeQueue results) {
    return NeighborArray.drainResults(results);
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
