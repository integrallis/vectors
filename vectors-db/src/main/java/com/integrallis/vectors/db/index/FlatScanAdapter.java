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
package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

/**
 * Brute-force reference implementation of {@link IndexSpi}. Scores every stored vector against the
 * query and returns the top-{@code k}. Serves as the always-available backend and as the
 * ground-truth reference for graph-based backends.
 *
 * <p><b>Ignored parameters.</b> Because brute-force already examines every vector, both {@code
 * searchListSize} and {@code overQueryFactor} are <i>ignored</i>. Callers should not rely on them
 * to affect flat-scan output — any value produces the same result. This matches the parameter
 * contract documented on {@link IndexSpi#search(float[], int, int, float)}.
 *
 * <p>Build requires exclusive access. Concurrent searches require safe publication after build and
 * no concurrent rebuild. The collection facade publishes complete index generations through a
 * volatile reference.
 */
public final class FlatScanAdapter implements IndexSpi, ExactOrdinalScorer {

  // Shared across adapters: concurrent collections must also respect the CPU expansion budget.
  static final Semaphore PARALLEL_SCANS = new Semaphore(1);
  static final AtomicInteger CONCURRENT_BATCHES = new AtomicInteger();
  private static final QueryBudget QUERY_BUDGET = new QueryBudget(System::nanoTime);

  /** Avoid expanding the first query of every burst while callers supply their own parallelism. */
  static final class QueryBudget {
    private static final long QUIET_NANOS = 50_000_000L;
    private final AtomicInteger active = new AtomicInteger();
    private final LongSupplier clock;
    private volatile long lastOverlap;

    QueryBudget(LongSupplier clock) {
      this.clock = clock;
      lastOverlap = clock.getAsLong() - QUIET_NANOS;
    }

    boolean enter() {
      int count = active.incrementAndGet();
      long now = clock.getAsLong();
      if (count > 1) {
        lastOverlap = now;
        return false;
      }
      return now - lastOverlap >= QUIET_NANOS;
    }

    void exit() {
      if (active.getAndDecrement() > 1) lastOverlap = clock.getAsLong();
    }

    boolean contended() {
      return active.get() > 1;
    }
  }

  /**
   * Run a collection batch that already distributes queries across processors. The collection
   * facade owns this scope; callers do not need to select a scheduling mode. A global counter
   * covers worker threads and concurrent collections, and is released even when a query fails.
   */
  public static <T> T withConcurrentQueries(Supplier<T> batch) {
    CONCURRENT_BATCHES.incrementAndGet();
    try {
      return batch.get();
    } finally {
      CONCURRENT_BATCHES.decrementAndGet();
    }
  }

  private float[][] vectors = new float[0][];
  private SimilarityFunction metric;
  private int dimension;

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.vectors = vectors;
    this.metric = metric;
    this.dimension = vectors.length == 0 ? 0 : vectors[0].length;
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (vectors.length == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }

    // COSINE scans are qualified for expansion; other metrics keep the original serial loop.
    // Batches already distribute queries across processors and do not acquire this budget.
    boolean budgeted =
        metric == SimilarityFunction.COSINE
            && (long) vectors.length * dimension >= 4_000_000L
            && ForkJoinPool.getCommonPoolParallelism() > 1
            && CONCURRENT_BATCHES.get() == 0;
    boolean isolated = budgeted && QUERY_BUDGET.enter();
    try {
      if (isolated && CONCURRENT_BATCHES.get() == 0 && PARALLEL_SCANS.tryAcquire()) {
        try {
          return selectScores(scoreAll(query, vectors, metric), k);
        } finally {
          PARALLEL_SCANS.release();
        }
      }
      int actualK = Math.min(k, vectors.length);

      // Bounded min-heap (by score) over at most actualK entries. When full, the root is the
      // worst-so-far kept result; a new candidate with strictly higher score replaces the root.
      int[] heapIds = new int[actualK];
      float[] heapScores = new float[actualK];
      int heapSize = 0;

      for (int i = 0; i < vectors.length; i++) {
        float score = metric.compare(query, vectors[i]);
        if (heapSize < actualK) {
          heapIds[heapSize] = i;
          heapScores[heapSize] = score;
          heapSize++;
          siftUp(heapIds, heapScores, heapSize - 1);
        } else if (score > heapScores[0]) {
          heapIds[0] = i;
          heapScores[0] = score;
          siftDown(heapIds, heapScores, 0, heapSize);
        }
      }

      // Drain heap into a descending-sorted result array.
      int[] sortedIds = new int[heapSize];
      float[] sortedScores = new float[heapSize];
      for (int i = heapSize - 1; i >= 0; i--) {
        sortedIds[i] = heapIds[0];
        sortedScores[i] = heapScores[0];
        heapIds[0] = heapIds[i];
        heapScores[0] = heapScores[i];
        siftDown(heapIds, heapScores, 0, i);
      }
      return new SearchOutcome(sortedIds, sortedScores);
    } finally {
      if (budgeted) QUERY_BUDGET.exit();
    }
  }

  /** Score independent rows using a bounded shared pool; never cache caller-owned vector data. */
  static float[] scoreAll(float[] query, float[][] rows, SimilarityFunction metric) {
    float[] scores = new float[rows.length];
    int workers = Math.min(8, ForkJoinPool.getCommonPoolParallelism());
    int chunk = Math.max(1, (rows.length - 1) / workers + 1);
    ForkJoinPool.commonPool()
        .invoke(new ScoreTask(query, rows, metric, scores, 0, rows.length, chunk));
    return scores;
  }

  private static final class ScoreTask extends RecursiveAction {
    @java.io.Serial private static final long serialVersionUID = 1L;
    private final float[] query;
    private final float[][] rows;
    private final SimilarityFunction metric;
    private final float[] scores;
    private final int from, to, chunk;

    ScoreTask(
        float[] query,
        float[][] rows,
        SimilarityFunction metric,
        float[] scores,
        int from,
        int to,
        int chunk) {
      this.query = query;
      this.rows = rows;
      this.metric = metric;
      this.scores = scores;
      this.from = from;
      this.to = to;
      this.chunk = chunk;
    }

    @Override
    protected void compute() {
      if (to - from <= chunk || QUERY_BUDGET.contended() || CONCURRENT_BATCHES.get() != 0) {
        for (int i = from; i < to; i++) scores[i] = metric.compare(query, rows[i]);
      } else {
        int middle = (from + to) >>> 1;
        var left = new ScoreTask(query, rows, metric, scores, from, middle, chunk);
        var right = new ScoreTask(query, rows, metric, scores, middle, to, chunk);
        left.fork();
        try {
          right.compute();
        } finally {
          // Join even on failure: no worker may retain query inputs after this call returns.
          left.join();
        }
      }
    }
  }

  /** Run the original bounded heap in the original row order, preserving ties exactly. */
  private SearchOutcome selectScores(float[] allScores, int k) {
    int actualK = Math.min(k, allScores.length);

    // Bounded min-heap (by score) over at most actualK entries. When full, the root is the
    // worst-so-far kept result; a new candidate with strictly higher score replaces the root.
    int[] heapIds = new int[actualK];
    float[] heapScores = new float[actualK];
    int heapSize = 0;

    for (int i = 0; i < allScores.length; i++) {
      float score = allScores[i];
      if (heapSize < actualK) {
        heapIds[heapSize] = i;
        heapScores[heapSize] = score;
        heapSize++;
        siftUp(heapIds, heapScores, heapSize - 1);
      } else if (score > heapScores[0]) {
        heapIds[0] = i;
        heapScores[0] = score;
        siftDown(heapIds, heapScores, 0, heapSize);
      }
    }

    // Drain heap into a descending-sorted result array.
    int[] sortedIds = new int[heapSize];
    float[] sortedScores = new float[heapSize];
    for (int i = heapSize - 1; i >= 0; i--) {
      sortedIds[i] = heapIds[0];
      sortedScores[i] = heapScores[0];
      heapIds[0] = heapIds[i];
      heapScores[0] = heapScores[i];
      siftDown(heapIds, heapScores, 0, i);
    }
    return new SearchOutcome(sortedIds, sortedScores);
  }

  /**
   * Batched brute-force scan. Performs a single pass over the corpus, updating a per-query bounded
   * min-heap as each stored vector is visited. Compared to running {@link #search} sequentially,
   * this amortises the outer loop over vectors and keeps each row hot in cache while every query
   * scores against it — a material speedup for cache-bound brute-force scans with several queries.
   */
  @Override
  public SearchOutcome[] searchBatch(
      float[][] queries, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(queries, "queries must not be null");
    if (queries.length == 0) {
      throw new IllegalArgumentException("queries must not be empty");
    }
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    int q = queries.length;
    SearchOutcome[] out = new SearchOutcome[q];
    if (vectors.length == 0) {
      for (int i = 0; i < q; i++) {
        out[i] = new SearchOutcome(new int[0], new float[0]);
      }
      return out;
    }
    // Validate dimensions up front so a late failure cannot corrupt partial results.
    for (int i = 0; i < q; i++) {
      Objects.requireNonNull(queries[i], "queries[" + i + "] must not be null");
      if (queries[i].length != dimension) {
        throw new IllegalArgumentException(
            "Query "
                + i
                + " dimension "
                + queries[i].length
                + " does not match index dimension "
                + dimension);
      }
    }
    int actualK = Math.min(k, vectors.length);
    int[][] heapIds = new int[q][actualK];
    float[][] heapScores = new float[q][actualK];
    int[] heapSizes = new int[q];
    if (metric == SimilarityFunction.COSINE) {
      // Small batches do not amortize the separate row-norm pass.
      if (q >= 4 && VectorUtil.supportsCosineNormReuse(dimension)) {
        scanCosineWithBothNorms(queries, actualK, heapIds, heapScores, heapSizes);
      } else {
        scanCosineBatch(queries, actualK, heapIds, heapScores, heapSizes);
      }
    } else {
      for (int v = 0; v < vectors.length; v++) {
        float[] stored = vectors[v];
        for (int qi = 0; qi < q; qi++) {
          float score = metric.compare(queries[qi], stored);
          int sz = heapSizes[qi];
          int[] ids = heapIds[qi];
          float[] scores = heapScores[qi];
          if (sz < actualK) {
            ids[sz] = v;
            scores[sz] = score;
            heapSizes[qi] = sz + 1;
            siftUp(ids, scores, sz);
          } else if (score > scores[0]) {
            ids[0] = v;
            scores[0] = score;
            siftDown(ids, scores, 0, sz);
          }
        }
      }
    }
    for (int qi = 0; qi < q; qi++) {
      int sz = heapSizes[qi];
      int[] sortedIds = new int[sz];
      float[] sortedScores = new float[sz];
      int[] ids = heapIds[qi];
      float[] scores = heapScores[qi];
      for (int i = sz - 1; i >= 0; i--) {
        sortedIds[i] = ids[0];
        sortedScores[i] = scores[0];
        ids[0] = ids[i];
        scores[0] = scores[i];
        siftDown(ids, scores, 0, i);
      }
      out[qi] = new SearchOutcome(sortedIds, sortedScores);
    }
    return out;
  }

  /** Keep cosine preparation and dispatch out of the other metrics' inner scan loops. */
  private void scanCosineBatch(
      float[][] queries, int actualK, int[][] heapIds, float[][] heapScores, int[] heapSizes) {
    float[] queryNorms = new float[queries.length];
    for (int qi = 0; qi < queries.length; qi++) {
      queryNorms[qi] = VectorUtil.cosineQueryNorm(queries[qi]);
    }
    for (int v = 0; v < vectors.length; v++) {
      float[] stored = vectors[v];
      for (int qi = 0; qi < queries.length; qi++) {
        float score =
            (1f + VectorUtil.cosineWithQueryNorm(queries[qi], stored, queryNorms[qi])) / 2f;
        int sz = heapSizes[qi];
        int[] ids = heapIds[qi];
        float[] scores = heapScores[qi];
        if (sz < actualK) {
          ids[sz] = v;
          scores[sz] = score;
          heapSizes[qi] = sz + 1;
          siftUp(ids, scores, sz);
        } else if (score > scores[0]) {
          ids[0] = v;
          scores[0] = score;
          siftDown(ids, scores, 0, sz);
        }
      }
    }
  }

  private void scanCosineWithBothNorms(
      float[][] queries, int actualK, int[][] heapIds, float[][] heapScores, int[] heapSizes) {
    float[] queryNorms = new float[queries.length];
    for (int qi = 0; qi < queries.length; qi++) {
      queryNorms[qi] = VectorUtil.cosineQueryNorm(queries[qi]);
    }
    for (int v = 0; v < vectors.length; v++) {
      float[] stored = vectors[v];
      // Rows are mutable: reuse only within this scan, never across calls.
      float rowNorm = VectorUtil.cosineQueryNorm(stored);
      for (int qi = 0; qi < queries.length; qi++) {
        float score =
            (1f + VectorUtil.cosineWithNorms(queries[qi], stored, queryNorms[qi], rowNorm)) / 2f;
        int sz = heapSizes[qi];
        int[] ids = heapIds[qi];
        float[] scores = heapScores[qi];
        if (sz < actualK) {
          ids[sz] = v;
          scores[sz] = score;
          heapSizes[qi] = sz + 1;
          siftUp(ids, scores, sz);
        } else if (score > scores[0]) {
          ids[0] = v;
          scores[0] = score;
          siftDown(ids, scores, 0, sz);
        }
      }
    }
  }

  @Override
  public int size() {
    return vectors.length;
  }

  @Override
  public OrdinalScorer exactScorerFor(float[] query) {
    Objects.requireNonNull(query, "query must not be null");
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    return ordinal -> metric.compare(query, vectors[ordinal]);
  }

  /** Sifts the element at {@code idx} up the min-heap to restore the heap invariant. */
  private static void siftUp(int[] ids, float[] scores, int idx) {
    while (idx > 0) {
      int parent = (idx - 1) >>> 1;
      if (scores[parent] <= scores[idx]) {
        break;
      }
      float tmpScore = scores[parent];
      int tmpId = ids[parent];
      scores[parent] = scores[idx];
      ids[parent] = ids[idx];
      scores[idx] = tmpScore;
      ids[idx] = tmpId;
      idx = parent;
    }
  }

  /** Sifts the element at {@code idx} down the min-heap to restore the heap invariant. */
  private static void siftDown(int[] ids, float[] scores, int idx, int size) {
    while (true) {
      int left = (idx << 1) + 1;
      int right = left + 1;
      int smallest = idx;
      if (left < size && scores[left] < scores[smallest]) {
        smallest = left;
      }
      if (right < size && scores[right] < scores[smallest]) {
        smallest = right;
      }
      if (smallest == idx) {
        break;
      }
      float tmpScore = scores[smallest];
      int tmpId = ids[smallest];
      scores[smallest] = scores[idx];
      ids[smallest] = ids[idx];
      scores[idx] = tmpScore;
      ids[idx] = tmpId;
      idx = smallest;
    }
  }
}
