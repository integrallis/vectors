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
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ScoreFunction;
import java.util.Objects;

/**
 * Flat scan variant that uses compressed vectors to select a candidate pool and then rescans that
 * pool with full-precision vectors. Final scores are exact for the selected candidates.
 */
public final class QuantizedFlatScanAdapter implements IndexSpi {

  private final IndexSpi exact;
  private final ExactOrdinalScorer exactScorer;
  private final SimilarityFunction metric;
  private final CompressedVectors compressed;

  public QuantizedFlatScanAdapter(
      IndexSpi exact,
      ExactOrdinalScorer exactScorer,
      SimilarityFunction metric,
      CompressedVectors compressed) {
    this.exact = Objects.requireNonNull(exact, "exact");
    this.exactScorer = Objects.requireNonNull(exactScorer, "exactScorer");
    this.metric = Objects.requireNonNull(metric, "metric");
    this.compressed = Objects.requireNonNull(compressed, "compressed");
    if (exact.size() != compressed.size()) {
      throw new IllegalArgumentException(
          "compressed size differs from flat index size: "
              + compressed.size()
              + " != "
              + exact.size());
    }
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "QuantizedFlatScanAdapter wraps an already-built flat index");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (compressed.size() == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (overQueryFactor <= 1.0f) {
      return exact.search(query, k, searchListSize, overQueryFactor);
    }

    int candidateK = Math.min(compressed.size(), Math.max(k, (int) Math.ceil(k * overQueryFactor)));
    int[] candidateIds = new int[candidateK];
    float[] candidateScores = new float[candidateK];
    int candidateSize = selectApproximateCandidates(query, candidateIds, candidateScores);

    ExactOrdinalScorer.OrdinalScorer scorer = exactScorer.exactScorerFor(query);
    int actualK = Math.min(k, candidateSize);
    int[] heapIds = new int[actualK];
    float[] heapScores = new float[actualK];
    int heapSize = 0;
    for (int i = 0; i < candidateSize; i++) {
      int ordinal = candidateIds[i];
      float score = scorer.score(ordinal);
      if (heapSize < actualK) {
        heapIds[heapSize] = ordinal;
        heapScores[heapSize] = score;
        siftUp(heapIds, heapScores, heapSize++);
      } else if (score > heapScores[0]) {
        heapIds[0] = ordinal;
        heapScores[0] = score;
        siftDown(heapIds, heapScores, 0, heapSize);
      }
    }
    return drain(heapIds, heapScores, heapSize);
  }

  private int selectApproximateCandidates(float[] query, int[] heapIds, float[] heapScores) {
    ScoreFunction approximate = compressed.scoreFunctionFor(query, metric);
    int heapSize = 0;
    for (int ordinal = 0; ordinal < compressed.size(); ordinal++) {
      float score = approximate.score(ordinal);
      if (heapSize < heapIds.length) {
        heapIds[heapSize] = ordinal;
        heapScores[heapSize] = score;
        siftUp(heapIds, heapScores, heapSize++);
      } else if (score > heapScores[0]) {
        heapIds[0] = ordinal;
        heapScores[0] = score;
        siftDown(heapIds, heapScores, 0, heapSize);
      }
    }
    return heapSize;
  }

  @Override
  public int size() {
    return exact.size();
  }

  @Override
  public void close() {
    exact.close();
    compressed.close();
  }

  private static SearchOutcome drain(int[] heapIds, float[] heapScores, int heapSize) {
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

  private static void siftUp(int[] ids, float[] scores, int idx) {
    while (idx > 0) {
      int parent = (idx - 1) >>> 1;
      if (scores[parent] <= scores[idx]) {
        break;
      }
      swap(ids, scores, parent, idx);
      idx = parent;
    }
  }

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
      swap(ids, scores, smallest, idx);
      idx = smallest;
    }
  }

  private static void swap(int[] ids, float[] scores, int a, int b) {
    float score = scores[a];
    int id = ids[a];
    scores[a] = scores[b];
    ids[a] = ids[b];
    scores[b] = score;
    ids[b] = id;
  }
}
