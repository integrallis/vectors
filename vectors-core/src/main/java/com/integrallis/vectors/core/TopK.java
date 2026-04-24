package com.integrallis.vectors.core;

import java.util.Objects;

/**
 * Selects the top-{@code k} ordinals from a dense score array by highest score.
 *
 * <p>Typical usage is in-VM rescoring after a coarse retrieval step (e.g. a network call to
 * Pinecone or Qdrant returns the top-1000 candidate ordinals; {@link VectorUtil#batchDotProduct}
 * computes exact float32 scores; {@link #select(float[], int)} picks the final top-k):
 *
 * <pre>{@code
 * float[] scores = new float[candidateVectors.length];
 * VectorUtil.batchDotProduct(query, candidateVectors, scores);
 * int[] topK = TopK.select(scores, k);
 * }</pre>
 *
 * <p>The returned ordinals are sorted by score descending. Ties are broken by ordinal ascending —
 * the lower ordinal wins — so the result is deterministic for equal scores.
 *
 * <p>Runs in {@code O(n log k)} using a size-{@code k} min-heap, with a single pass over the score
 * array and no boxing. For {@code k >= scores.length} this degenerates to a full descending sort.
 *
 * <p>This utility assumes "higher is better" — the convention used by {@link SimilarityFunction}
 * for all four built-in metrics. Callers scoring with raw squared L2 (lower is better) should
 * negate before calling.
 */
public final class TopK {

  private TopK() {}

  /**
   * Returns the ordinals of the top-{@code k} entries of {@code scores} by highest score, sorted
   * descending. Ties are broken by ordinal ascending.
   *
   * @param scores dense score array; not null
   * @param k number of ordinals to return; clamped to {@code [0, scores.length]}
   * @return a fresh {@code int[]} of length {@code min(k, scores.length)}; empty if {@code k <= 0}
   *     or {@code scores.length == 0}
   * @throws NullPointerException if {@code scores} is null
   * @throws IllegalArgumentException if any score is {@code NaN}
   */
  public static int[] select(float[] scores, int k) {
    Objects.requireNonNull(scores, "scores must not be null");
    int n = scores.length;
    if (k <= 0 || n == 0) {
      return new int[0];
    }
    int actualK = Math.min(k, n);

    // Min-heap of size actualK keyed on (score asc, ordinal desc) so the root is the worst element
    // still in the top-k — the next candidate to evict. Parallel id[] / score[] arrays avoid any
    // boxing or comparator allocation.
    int[] heapIds = new int[actualK];
    float[] heapScores = new float[actualK];

    for (int i = 0; i < n; i++) {
      float s = scores[i];
      if (Float.isNaN(s)) {
        throw new IllegalArgumentException("NaN score at ordinal " + i);
      }
      if (i < actualK) {
        heapIds[i] = i;
        heapScores[i] = s;
        siftUp(heapIds, heapScores, i);
      } else if (worseThanRoot(s, i, heapScores[0], heapIds[0])) {
        // New candidate is not better than the current root — skip.
      } else {
        heapIds[0] = i;
        heapScores[0] = s;
        siftDown(heapIds, heapScores, 0, actualK);
      }
    }

    // Heap-sort the result into descending order by repeatedly popping the root (worst) into the
    // tail of the output.
    int[] out = new int[actualK];
    for (int i = actualK - 1; i >= 0; i--) {
      out[i] = heapIds[0];
      heapIds[0] = heapIds[i];
      heapScores[0] = heapScores[i];
      siftDown(heapIds, heapScores, 0, i);
    }
    return out;
  }

  /** True iff candidate (s, ord) is no better than the root (rootScore, rootOrd). */
  private static boolean worseThanRoot(float s, int ord, float rootScore, int rootOrd) {
    if (s < rootScore) return true;
    if (s > rootScore) return false;
    // Equal scores: lower ordinal is better, so a higher ordinal is worse.
    return ord > rootOrd;
  }

  /** True iff heap element a is "less" (more evictable) than b under (score asc, ordinal desc). */
  private static boolean less(float[] scores, int[] ids, int a, int b) {
    float sa = scores[a];
    float sb = scores[b];
    if (sa != sb) return sa < sb;
    return ids[a] > ids[b];
  }

  private static void siftUp(int[] ids, float[] scores, int idx) {
    while (idx > 0) {
      int parent = (idx - 1) >>> 1;
      if (!less(scores, ids, idx, parent)) break;
      swap(ids, scores, idx, parent);
      idx = parent;
    }
  }

  private static void siftDown(int[] ids, float[] scores, int idx, int size) {
    while (true) {
      int left = (idx << 1) + 1;
      int right = left + 1;
      int smallest = idx;
      if (left < size && less(scores, ids, left, smallest)) smallest = left;
      if (right < size && less(scores, ids, right, smallest)) smallest = right;
      if (smallest == idx) break;
      swap(ids, scores, idx, smallest);
      idx = smallest;
    }
  }

  private static void swap(int[] ids, float[] scores, int i, int j) {
    int tmpId = ids[i];
    ids[i] = ids[j];
    ids[j] = tmpId;
    float tmpScore = scores[i];
    scores[i] = scores[j];
    scores[j] = tmpScore;
  }
}
