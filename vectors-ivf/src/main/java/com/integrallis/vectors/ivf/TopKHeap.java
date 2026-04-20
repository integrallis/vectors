package com.integrallis.vectors.ivf;

/**
 * Bounded top-K min-heap over {@code (ordinal, score)} pairs backed by primitive arrays. Retains
 * the {@code k} highest scores seen. Zero per-entry allocation; designed for tight inner scan loops
 * in IVF and similar flat-scan code paths.
 *
 * <p>The heap invariant keeps the <em>smallest retained score</em> at index 0, so a candidate is
 * admitted iff its score beats {@link #worst()} once the heap is saturated.
 *
 * <p>Not thread-safe; instances are meant to be scoped to a single search call.
 */
final class TopKHeap {

  private final int capacity;
  private final int[] ordinals;
  private final float[] scores;
  private int size;

  TopKHeap(int capacity) {
    if (capacity <= 0) throw new IllegalArgumentException("capacity must be > 0: " + capacity);
    this.capacity = capacity;
    this.ordinals = new int[capacity];
    this.scores = new float[capacity];
    this.size = 0;
  }

  int size() {
    return size;
  }

  boolean isFull() {
    return size == capacity;
  }

  /** Smallest retained score, or {@code Float.NEGATIVE_INFINITY} when empty. */
  float worst() {
    return size == 0 ? Float.NEGATIVE_INFINITY : scores[0];
  }

  /**
   * Offers {@code (ordinal, score)}. Returns {@code true} if it was admitted (i.e. heap grew or
   * replaced the previous worst). When saturated, rejects scores that do not beat {@link #worst()}.
   */
  boolean offer(int ordinal, float score) {
    if (size < capacity) {
      ordinals[size] = ordinal;
      scores[size] = score;
      size++;
      siftUp(size - 1);
      return true;
    }
    if (score <= scores[0]) return false;
    ordinals[0] = ordinal;
    scores[0] = score;
    siftDown(0);
    return true;
  }

  /**
   * Drains the heap into descending-score order in-place, returning parallel arrays {@code
   * (ordinals, scores)} of length {@link #size()}. Subsequent heap operations are undefined;
   * intended as a terminal operation.
   */
  DrainResult drainDescending() {
    int n = size;
    int[] outOrds = new int[n];
    float[] outScores = new float[n];
    for (int i = n - 1; i >= 0; i--) {
      outOrds[i] = ordinals[0];
      outScores[i] = scores[0];
      size--;
      if (size > 0) {
        ordinals[0] = ordinals[size];
        scores[0] = scores[size];
        siftDown(0);
      }
    }
    return new DrainResult(outOrds, outScores);
  }

  private void siftUp(int i) {
    while (i > 0) {
      int parent = (i - 1) >>> 1;
      if (scores[i] < scores[parent]) {
        swap(i, parent);
        i = parent;
      } else {
        return;
      }
    }
  }

  private void siftDown(int i) {
    int half = size >>> 1;
    while (i < half) {
      int left = (i << 1) + 1;
      int right = left + 1;
      int smallest = left;
      if (right < size && scores[right] < scores[left]) smallest = right;
      if (scores[i] <= scores[smallest]) return;
      swap(i, smallest);
      i = smallest;
    }
  }

  private void swap(int a, int b) {
    int ti = ordinals[a];
    ordinals[a] = ordinals[b];
    ordinals[b] = ti;
    float tf = scores[a];
    scores[a] = scores[b];
    scores[b] = tf;
  }

  /** Parallel arrays returned by {@link #drainDescending()}. */
  record DrainResult(int[] ordinals, float[] scores) {}
}
