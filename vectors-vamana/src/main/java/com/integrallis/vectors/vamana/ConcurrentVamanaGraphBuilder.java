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
package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Multi-threaded Vamana graph builder implementing a concurrent variant of the DiskANN two-pass
 * construction algorithm.
 *
 * <p>Build phases (same logical flow as {@link VamanaGraphBuilder}):
 *
 * <ol>
 *   <li><b>Sequential</b>: initialize nodes, compute medoid, build random graph.
 *   <li><b>Concurrent pass 1</b> (alpha=1.0): partition the random node permutation across threads;
 *       each thread processes its subset, using per-node {@link ReentrantLock}s for neighbor-list
 *       reads and writes.
 *   <li><b>Concurrent pass 2</b> (configured alpha): same as pass 1, on the improved graph.
 * </ol>
 *
 * <p>Because the algorithm is approximate, threads that read a neighbor list slightly ahead of a
 * concurrent write simply see a momentarily stale view — the second pass corrects any degradation.
 * Writes (forward replacement + backlinks) always hold the target node's lock for the full
 * insert-and-prune operation, preventing data corruption.
 */
public final class ConcurrentVamanaGraphBuilder {

  private final int maxDegree;
  private final int searchListSize;
  private final float alpha;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction sim;
  private final Random random;

  private ConcurrentVamanaGraphBuilder(
      int maxDegree,
      int searchListSize,
      float alpha,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      long seed) {
    this.maxDegree = maxDegree;
    this.searchListSize = searchListSize;
    this.alpha = alpha;
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
    this.sim = Objects.requireNonNull(sim, "sim must not be null");
    this.random = new Random(seed);
  }

  /**
   * Creates a new concurrent builder.
   *
   * @param maxDegree maximum degree R
   * @param searchListSize beam width L
   * @param alpha diversity parameter (>= 1.0)
   * @param vectors the dataset
   * @param sim similarity function
   * @param seed random seed (deterministic permutation)
   */
  public static ConcurrentVamanaGraphBuilder create(
      int maxDegree,
      int searchListSize,
      float alpha,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      long seed) {
    if (maxDegree <= 0)
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    if (searchListSize <= 0)
      throw new IllegalArgumentException("searchListSize must be positive: " + searchListSize);
    if (alpha < 1.0f) throw new IllegalArgumentException("alpha must be >= 1.0: " + alpha);
    return new ConcurrentVamanaGraphBuilder(maxDegree, searchListSize, alpha, vectors, sim, seed);
  }

  /** Builds using {@code Runtime.availableProcessors()} threads. */
  public VamanaGraph build() {
    return build(Runtime.getRuntime().availableProcessors());
  }

  /**
   * Builds the Vamana graph using {@code parallelism} threads.
   *
   * @param parallelism number of worker threads
   * @return the completed graph
   */
  public VamanaGraph build(int parallelism) {
    int n = vectors.size();
    var graph = new VamanaGraph(n, maxDegree);
    var locks = new ReentrantLock[n];
    for (int i = 0; i < n; i++) {
      graph.initNode(i);
      locks[i] = new ReentrantLock();
    }

    int medoid = computeMedoid();
    graph.setMedoid(medoid);

    initializeRandomGraph(graph);

    int[] order1 = makePermutation(n);
    buildPassConcurrent(graph, locks, 1.0f, order1, parallelism);

    if (alpha > 1.0f) {
      int[] order2 = makePermutation(n);
      buildPassConcurrent(graph, locks, alpha, order2, parallelism);
    }

    return graph;
  }

  // ---------------------------------------------------------------------------
  // Per-thread scratch buffers
  // ---------------------------------------------------------------------------

  private static final class WorkContext {
    final BitSet visited;
    final NodeQueue candidates;
    final NodeQueue results;
    final int[] tmpIds;
    final NeighborArray pruneCandidates;
    final NeighborArray pruneResult;
    final NeighborArray backlinkCandidates;
    final NeighborArray backlinkResult;
    // Per-thread stable copy of the current query. A shared-buffer store overwrites its returned
    // array on every getVector() call, so `query` must be copied here before the beam search (which
    // calls getVector() repeatedly) aliases it. Null for stable float[][] stores. Each thread owns
    // its own WorkContext, so this row is never shared across threads.
    final float[] queryScratch;

    WorkContext(int maxNodes, int searchListSize, int maxDegree, int dim, boolean sharedBuffer) {
      visited = new BitSet(maxNodes);
      candidates = new NodeQueue(Math.max(64, searchListSize * 2), false);
      results = new NodeQueue(Math.max(64, searchListSize * 2), true);
      tmpIds = new int[maxDegree + 1];
      pruneCandidates = new NeighborArray(searchListSize + maxDegree + 1);
      pruneResult = new NeighborArray(maxDegree + 1);
      backlinkCandidates = new NeighborArray(maxDegree + 1);
      backlinkResult = new NeighborArray(maxDegree + 1);
      queryScratch = sharedBuffer ? new float[dim] : null;
    }
  }

  // ---------------------------------------------------------------------------
  // Concurrent pass
  // ---------------------------------------------------------------------------

  private void buildPassConcurrent(
      VamanaGraph graph, ReentrantLock[] locks, float passAlpha, int[] order, int parallelism) {
    int n = order.length;
    int chunkSize = (n + parallelism - 1) / parallelism;

    try (ExecutorService exec = Executors.newFixedThreadPool(parallelism)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < parallelism; t++) {
        int from = t * chunkSize;
        int to = Math.min((t + 1) * chunkSize, n);
        if (from >= n) break;
        final int[] chunk = Arrays.copyOfRange(order, from, to);
        futures.add(
            exec.submit(
                () -> {
                  var ctx =
                      new WorkContext(
                          vectors.size(),
                          searchListSize,
                          maxDegree,
                          vectors.dimension(),
                          vectors.sharesReturnBuffer());
                  for (int node : chunk) {
                    processNodeConcurrent(node, graph, locks, passAlpha, ctx);
                  }
                }));
      }
      awaitAll(futures);
    }
  }

  private void processNodeConcurrent(
      int node, VamanaGraph graph, ReentrantLock[] locks, float passAlpha, WorkContext ctx) {
    float[] query = vectors.getVector(node);
    if (ctx.queryScratch != null) {
      // Shared-buffer store: stabilize the query before searchConcurrent/getVector aliases it.
      System.arraycopy(query, 0, ctx.queryScratch, 0, ctx.queryScratch.length);
      query = ctx.queryScratch;
    }

    // Beam search from medoid — reads under per-node locks
    SearchResult searchResult = searchConcurrent(query, graph.medoid(), graph, locks, ctx);

    // Merge search results + existing neighbors into candidate set
    ctx.pruneCandidates.clear();
    for (int i = 0; i < searchResult.size(); i++) {
      int cand = searchResult.nodeId(i);
      if (cand != node) ctx.pruneCandidates.addUnsorted(cand, searchResult.score(i));
    }

    // Read existing neighbors under lock for safe snapshot
    locks[node].lock();
    try {
      NeighborArray existing = graph.getNeighbors(node);
      for (int i = 0; i < existing.size(); i++) {
        int nbr = existing.node(i);
        if (nbr != node && !ctx.pruneCandidates.contains(nbr)) {
          ctx.pruneCandidates.addUnsorted(nbr, sim.compare(query, vectors.getVector(nbr)));
        }
      }
    } finally {
      locks[node].unlock();
    }
    ctx.pruneCandidates.sort();

    // Robust prune → store result in ctx.pruneResult
    RobustPruner.robustPrune(
        node, ctx.pruneCandidates, maxDegree, passAlpha, vectors, sim, ctx.pruneResult);

    // Replace node's neighbor list (lock own node)
    locks[node].lock();
    try {
      NeighborArray existing = graph.getNeighbors(node);
      existing.clear();
      existing.copyFrom(ctx.pruneResult);
    } finally {
      locks[node].unlock();
    }

    // Add backlinks to each new neighbor
    for (int i = 0; i < ctx.pruneResult.size(); i++) {
      int neighbor = ctx.pruneResult.node(i);
      // Reuse the pruned score: the metric is symmetric, so compare(neighbor, query) is
      // bit-identical to the score already carried in pruneResult (compare(query, neighbor)).
      float backlinkScore = ctx.pruneResult.score(i);
      locks[neighbor].lock();
      try {
        NeighborArray nNbrs = graph.getNeighbors(neighbor);
        if (!nNbrs.contains(node)) {
          if (nNbrs.size() < maxDegree) {
            nNbrs.insert(node, backlinkScore);
          } else {
            ctx.backlinkCandidates.clear();
            for (int j = 0; j < nNbrs.size(); j++) {
              ctx.backlinkCandidates.addUnsorted(nNbrs.node(j), nNbrs.score(j));
            }
            ctx.backlinkCandidates.addUnsorted(node, backlinkScore);
            ctx.backlinkCandidates.sort();
            RobustPruner.robustPrune(
                neighbor,
                ctx.backlinkCandidates,
                maxDegree,
                passAlpha,
                vectors,
                sim,
                ctx.backlinkResult);
            nNbrs.clear();
            nNbrs.copyFrom(ctx.backlinkResult);
          }
        }
      } finally {
        locks[neighbor].unlock();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Concurrent beam search (reads neighbor lists under per-node locks)
  // ---------------------------------------------------------------------------

  private SearchResult searchConcurrent(
      float[] query, int entryPoint, VamanaGraph graph, ReentrantLock[] locks, WorkContext ctx) {
    int L = searchListSize;
    ctx.candidates.clear();
    ctx.results.clear();
    ctx.visited.clear();

    float entryScore = sim.compare(query, vectors.getVector(entryPoint));
    ctx.candidates.add(entryPoint, entryScore);
    ctx.results.add(entryPoint, entryScore);
    ctx.visited.set(entryPoint);

    while (!ctx.candidates.isEmpty()) {
      long top = ctx.candidates.poll();
      float candScore = NodeQueue.score(top);
      int candId = NodeQueue.nodeId(top);

      if (ctx.results.size() >= L && candScore < NodeQueue.score(ctx.results.peek())) break;

      // Snapshot neighbors under lock
      int nCount = snapshotNeighbors(candId, graph, locks, ctx.tmpIds);
      for (int i = 0; i < nCount; i++) {
        int nbr = ctx.tmpIds[i];
        if (ctx.visited.get(nbr)) continue;
        ctx.visited.set(nbr);
        float score = sim.compare(query, vectors.getVector(nbr));
        // Single sift-down eviction; only explore the neighbor if it entered the result beam.
        if (ctx.results.insertWithOverflow(nbr, score, L)) {
          ctx.candidates.add(nbr, score);
        }
      }
    }

    // Drain min-heap to descending arrays
    int sz = ctx.results.size();
    int[] nodeIds = new int[sz];
    float[] scores = new float[sz];
    for (int i = sz - 1; i >= 0; i--) {
      long e = ctx.results.poll();
      nodeIds[i] = NodeQueue.nodeId(e);
      scores[i] = NodeQueue.score(e);
    }
    return new SearchResult(nodeIds, scores);
  }

  private int snapshotNeighbors(int nodeId, VamanaGraph graph, ReentrantLock[] locks, int[] dest) {
    locks[nodeId].lock();
    try {
      NeighborArray na = graph.getNeighbors(nodeId);
      int n = na.size();
      for (int i = 0; i < n; i++) dest[i] = na.node(i);
      return n;
    } finally {
      locks[nodeId].unlock();
    }
  }

  // ---------------------------------------------------------------------------
  // Medoid, random init, permutation — same logic as VamanaGraphBuilder
  // ---------------------------------------------------------------------------

  /** Package-private so tests can call it directly (mirrors VamanaGraphBuilder.computeMedoid). */
  int computeMedoid() {
    int n = vectors.size();
    int dim = vectors.dimension();
    float[] centroid = new float[dim];
    for (int i = 0; i < n; i++) {
      float[] v = vectors.getVector(i);
      for (int d = 0; d < dim; d++) centroid[d] += v[d];
    }
    for (int d = 0; d < dim; d++) centroid[d] /= n;

    int best = 0;
    float bestDist = Float.MAX_VALUE;
    for (int i = 0; i < n; i++) {
      float[] v = vectors.getVector(i);
      float dist = 0f;
      for (int d = 0; d < dim; d++) {
        float diff = v[d] - centroid[d];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  private void initializeRandomGraph(VamanaGraph graph) {
    int n = vectors.size();
    int numRandom = Math.min(maxDegree, n - 1);
    int[] candidates = n > 1 ? new int[n - 1] : new int[0];
    // See VamanaGraphBuilder.initializeRandomGraph: on a shared-buffer store the two getVector()
    // calls alias the same array, so compare(getVector(node), getVector(nbr)) == compare(x, x) == 0
    // for every seed edge. Stabilize the node vector in a scratch row first.
    boolean sharedBuffer = vectors.sharesReturnBuffer();
    float[] nodeScratch = sharedBuffer ? new float[vectors.dimension()] : null;

    for (int node = 0; node < n; node++) {
      var neighbors = graph.getNeighbors(node);
      int idx = 0;
      for (int i = 0; i < n; i++) if (i != node) candidates[idx++] = i;
      float[] nodeVec = vectors.getVector(node);
      if (sharedBuffer) {
        System.arraycopy(nodeVec, 0, nodeScratch, 0, nodeScratch.length);
        nodeVec = nodeScratch;
      }
      for (int i = 0; i < numRandom; i++) {
        int j = i + random.nextInt(candidates.length - i);
        int tmp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = tmp;
        int nbr = candidates[i];
        float score = sim.compare(nodeVec, vectors.getVector(nbr));
        neighbors.insert(nbr, score);
      }
    }
  }

  private int[] makePermutation(int n) {
    int[] order = new int[n];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    return order;
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
