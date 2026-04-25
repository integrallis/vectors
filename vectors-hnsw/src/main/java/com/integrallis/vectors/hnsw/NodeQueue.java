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

/**
 * Long-encoded binary heap for beam search. Each entry packs (score, nodeId) into a single {@code
 * long}: upper 32 bits = {@link Float#floatToRawIntBits(float) floatToRawIntBits(score)}, lower 32
 * bits = nodeId.
 *
 * <p>This encoding preserves score ordering for non-negative floats (all {@link
 * com.integrallis.vectors.core.SimilarityFunction} scores are non-negative), enabling single {@code
 * long} comparisons in heap operations with zero object allocation.
 *
 * <p>Two modes:
 *
 * <ul>
 *   <li><b>Min-heap</b> ({@code minHeap=true}): poll returns lowest score — used for result sets
 *       (worst result on top for bounded eviction).
 *   <li><b>Max-heap</b> ({@code minHeap=false}): poll returns highest score — used for candidate
 *       sets (best candidate on top for exploration).
 * </ul>
 */
public final class NodeQueue {

  private long[] heap;
  private int size;
  private final boolean minHeap;

  public NodeQueue(int initialCapacity, boolean minHeap) {
    this.heap = new long[Math.max(1, initialCapacity)];
    this.size = 0;
    this.minHeap = minHeap;
  }

  /** Adds a (nodeId, score) entry to the heap. */
  public void add(int nodeId, float score) {
    ensureCapacity(size + 1);
    heap[size] = encode(nodeId, score);
    siftUp(size);
    size++;
  }

  /** Removes and returns the top entry. */
  public long poll() {
    long top = heap[0];
    size--;
    if (size > 0) {
      heap[0] = heap[size];
      siftDown(0);
    }
    return top;
  }

  /** Returns the top entry without removing it. */
  public long peek() {
    return heap[0];
  }

  /** Returns the number of entries. */
  public int size() {
    return size;
  }

  /** Returns true if the queue is empty. */
  public boolean isEmpty() {
    return size == 0;
  }

  /** Removes all entries. */
  public void clear() {
    size = 0;
  }

  /**
   * Inserts a new entry if it improves the bounded result set.
   *
   * <p>For a min-heap result set (worst result on top): if the new score is better (higher) than
   * the worst result (top of min-heap), replaces it. For a max-heap candidate set: if the new score
   * is worse (lower) than the best candidate, replaces it.
   *
   * @return true if the entry was inserted
   */
  public boolean insertWithOverflow(int nodeId, float score, int bound) {
    if (size < bound) {
      add(nodeId, score);
      return true;
    }
    float topScore = score(heap[0]);
    if (minHeap) {
      // Min-heap: top is worst result (lowest score). Replace if new score is better.
      if (score > topScore) {
        heap[0] = encode(nodeId, score);
        siftDown(0);
        return true;
      }
    } else {
      // Max-heap: top is best candidate (highest score). Replace if new score is worse.
      if (score < topScore) {
        heap[0] = encode(nodeId, score);
        siftDown(0);
        return true;
      }
    }
    return false;
  }

  /**
   * Encodes a (nodeId, score) pair into a single long.
   *
   * <p>Requires {@code score >= 0}: negative floats set bit 63, which inverts {@link
   * Long#compareUnsigned} ordering and corrupts heap invariants. All {@link
   * com.integrallis.vectors.core.SimilarityFunction} implementations return non-negative scores.
   *
   * @throws IllegalArgumentException if score is negative or NaN
   */
  public static long encode(int nodeId, float score) {
    if (!(score >= 0f)) { // also catches NaN
      throw new IllegalArgumentException(
          "NodeQueue requires non-negative scores; got "
              + score
              + " for node "
              + nodeId
              + ". Negative scores corrupt the long-encoded heap ordering.");
    }
    return ((long) Float.floatToRawIntBits(score) << 32) | (nodeId & 0xFFFFFFFFL);
  }

  /** Extracts the nodeId from an encoded entry. */
  public static int nodeId(long entry) {
    return (int) entry;
  }

  /** Extracts the score from an encoded entry. */
  public static float score(long entry) {
    return Float.intBitsToFloat((int) (entry >>> 32));
  }

  private void ensureCapacity(int minCapacity) {
    if (minCapacity > heap.length) {
      int newCapacity = heap.length + (heap.length >> 1); // grow by 50%
      if (newCapacity < minCapacity) {
        newCapacity = minCapacity;
      }
      long[] newHeap = new long[newCapacity];
      System.arraycopy(heap, 0, newHeap, 0, size);
      heap = newHeap;
    }
  }

  private void siftUp(int index) {
    long value = heap[index];
    while (index > 0) {
      int parent = (index - 1) >>> 1;
      if (compare(value, heap[parent]) >= 0) {
        break;
      }
      heap[index] = heap[parent];
      index = parent;
    }
    heap[index] = value;
  }

  private void siftDown(int index) {
    long value = heap[index];
    int half = size >>> 1;
    while (index < half) {
      int child = (index << 1) + 1;
      long childVal = heap[child];
      int right = child + 1;
      if (right < size && compare(heap[right], childVal) < 0) {
        child = right;
        childVal = heap[right];
      }
      if (compare(value, childVal) <= 0) {
        break;
      }
      heap[index] = childVal;
      index = child;
    }
    heap[index] = value;
  }

  /**
   * Compares two encoded entries. For min-heap, natural long ordering (lower score first). For
   * max-heap, reverse ordering (higher score first).
   */
  private int compare(long a, long b) {
    // For non-negative floats, the upper 32-bit encoding preserves ordering.
    // Compare as unsigned longs to handle the full range correctly.
    int cmp = Long.compareUnsigned(a, b);
    return minHeap ? cmp : -cmp;
  }
}
