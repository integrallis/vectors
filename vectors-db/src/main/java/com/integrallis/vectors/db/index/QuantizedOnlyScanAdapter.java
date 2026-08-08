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
import java.util.function.IntPredicate;

/**
 * Flat scan over compressed vectors alone, with no full-precision copy to fall back on.
 *
 * <p>Distinct from {@link QuantizedFlatScanAdapter}, which ranks on the codes only to choose a
 * candidate pool and then rescores it exactly. That needs {@code vectors.bin} and therefore costs
 * more storage than not quantizing at all. Here the codes are all there is, so the scores this
 * returns are approximate and final.
 *
 * <p>The trade is worth making for an index that ships inside an artifact and is read far more
 * often than it is built. On a task classifier over EmbeddingGemma vectors, 4-bit codes cost 0.6
 * points of accuracy for an eighth of the bytes. It is a poor trade wherever exact scores matter,
 * which is why nothing selects it by default.
 */
public final class QuantizedOnlyScanAdapter implements IndexSpi {

  private final CompressedVectors compressed;
  private final SimilarityFunction metric;

  /**
   * Wraps already-decoded compressed vectors.
   *
   * @param compressed the codes, which are the whole index
   * @param metric how a query is scored against them
   */
  public QuantizedOnlyScanAdapter(CompressedVectors compressed, SimilarityFunction metric) {
    this.compressed = Objects.requireNonNull(compressed, "compressed");
    this.metric = Objects.requireNonNull(metric, "metric");
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "QuantizedOnlyScanAdapter wraps codes that were already encoded");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    return searchWithPredicate(query, k, searchListSize, overQueryFactor, ordinal -> true);
  }

  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    Objects.requireNonNull(query, "query must not be null");
    Objects.requireNonNull(predicate, "predicate must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    int size = compressed.size();
    if (size == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }

    // overQueryFactor is deliberately ignored: it exists to widen a candidate pool before an exact
    // rescore, and there is no rescore here. Honouring it would only cost time.
    ScoreFunction scorer = compressed.scoreFunctionFor(query, metric);
    int capacity = Math.min(k, size);
    int[] heapIds = new int[capacity];
    float[] heapScores = new float[capacity];
    int heapSize = 0;
    for (int ordinal = 0; ordinal < size; ordinal++) {
      if (!predicate.test(ordinal)) {
        continue;
      }
      float score = scorer.score(ordinal);
      if (heapSize < capacity) {
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

  @Override
  public int size() {
    return compressed.size();
  }

  @Override
  public void close() {
    compressed.close();
  }

  /** Min-heap sift-up, keeping the weakest surviving score at the root. */
  private static void siftUp(int[] ids, float[] scores, int index) {
    while (index > 0) {
      int parent = (index - 1) >>> 1;
      if (scores[parent] <= scores[index]) {
        break;
      }
      swap(ids, scores, parent, index);
      index = parent;
    }
  }

  private static void siftDown(int[] ids, float[] scores, int index, int size) {
    while (true) {
      int left = (index << 1) + 1;
      if (left >= size) {
        return;
      }
      int smallest = left;
      int right = left + 1;
      if (right < size && scores[right] < scores[left]) {
        smallest = right;
      }
      if (scores[index] <= scores[smallest]) {
        return;
      }
      swap(ids, scores, index, smallest);
      index = smallest;
    }
  }

  private static void swap(int[] ids, float[] scores, int left, int right) {
    int id = ids[left];
    ids[left] = ids[right];
    ids[right] = id;
    float score = scores[left];
    scores[left] = scores[right];
    scores[right] = score;
  }

  /** Empties the heap into descending score order, which is what callers expect. */
  private static SearchOutcome drain(int[] heapIds, float[] heapScores, int heapSize) {
    int[] ids = new int[heapSize];
    float[] scores = new float[heapSize];
    int remaining = heapSize;
    for (int position = heapSize - 1; position >= 0; position--) {
      ids[position] = heapIds[0];
      scores[position] = heapScores[0];
      remaining--;
      heapIds[0] = heapIds[remaining];
      heapScores[0] = heapScores[remaining];
      siftDown(heapIds, heapScores, 0, remaining);
    }
    return new SearchOutcome(ids, scores);
  }
}
