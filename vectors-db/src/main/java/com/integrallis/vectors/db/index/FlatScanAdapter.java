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
import java.util.Objects;

/**
 * Brute-force reference implementation of {@link IndexSpi}. Scores every stored vector against the
 * query and returns the top-{@code k}. Serves as the always-available backend (Step 2) and as the
 * ground-truth reference for the graph-based backends in later steps.
 *
 * <p><b>Ignored parameters.</b> Because brute-force already examines every vector, both {@code
 * searchListSize} and {@code overQueryFactor} are <i>ignored</i>. Callers should not rely on them
 * to affect flat-scan output — any value produces the same result. This matches the parameter
 * contract documented on {@link IndexSpi#search(float[], int, int, float)}.
 *
 * <p>Not thread-safe for concurrent {@link #build(float[][], SimilarityFunction)} calls; reads via
 * {@link #search(float[], int, int, float)} are safe as long as no build is in flight (the {@link
 * com.integrallis.vectors.db.VectorCollection} facade enforces this with a read/write lock).
 */
public final class FlatScanAdapter implements IndexSpi {

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

  @Override
  public int size() {
    return vectors.length;
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
