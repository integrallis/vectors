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

/**
 * Sorted parallel arrays of (nodeId, score) pairs, used for neighbor lists in Vamana graphs and as
 * scratch space during search and pruning.
 *
 * <p>Entries are maintained in <b>descending score order</b> (best first, worst at {@code size-1}).
 * This enables O(1) worst-neighbor access for eviction and ordered iteration for the diversity
 * heuristic.
 *
 * <p>For typical Vamana parameters (R=32-64), linear scan and {@link System#arraycopy} are faster
 * than tree-based structures due to cache locality.
 */
public final class NeighborArray {

  private final int[] nodes;
  private final float[] scores;
  private int size;

  /** Creates an empty array with the given maximum capacity. */
  public NeighborArray(int maxSize) {
    if (maxSize <= 0) {
      throw new IllegalArgumentException("maxSize must be positive: " + maxSize);
    }
    this.nodes = new int[maxSize];
    this.scores = new float[maxSize];
    this.size = 0;
  }

  /** Returns the number of entries currently stored. */
  public int size() {
    return size;
  }

  /** Returns the maximum capacity. */
  public int maxSize() {
    return nodes.length;
  }

  /** Returns the node ID at the given index (0 = best score). */
  public int node(int index) {
    return nodes[index];
  }

  /** Returns the score at the given index (0 = best score). */
  public float score(int index) {
    return scores[index];
  }

  /**
   * Inserts a (nodeId, score) pair, maintaining descending score order.
   *
   * <p>If the array is full and the new score is worse than or equal to the worst entry, the
   * insertion is rejected. If the array is full and the new score is better, the worst entry is
   * evicted. Duplicate nodeIds are rejected.
   *
   * @return true if inserted, false if rejected (duplicate or worse than worst when full)
   */
  public boolean insert(int nodeId, float score) {
    // Reject duplicates
    for (int i = 0; i < size; i++) {
      if (nodes[i] == nodeId) {
        return false;
      }
    }

    // Find insertion position via binary search (descending order)
    int pos = findInsertPosition(score);

    if (size < nodes.length) {
      // Not full: shift right and insert
      System.arraycopy(nodes, pos, nodes, pos + 1, size - pos);
      System.arraycopy(scores, pos, scores, pos + 1, size - pos);
      nodes[pos] = nodeId;
      scores[pos] = score;
      size++;
      return true;
    } else {
      // Full: reject if score <= worst (last element)
      if (score <= scores[size - 1]) {
        return false;
      }
      // Evict worst (last), shift right from pos, insert
      if (pos < size - 1) {
        System.arraycopy(nodes, pos, nodes, pos + 1, size - 1 - pos);
        System.arraycopy(scores, pos, scores, pos + 1, size - 1 - pos);
      }
      nodes[pos] = nodeId;
      scores[pos] = score;
      return true;
    }
  }

  /**
   * Adds a (nodeId, score) pair at the end without maintaining sort order. This is an O(1)
   * operation used during batch construction (e.g., collecting candidates for pruning).
   *
   * <p>The caller is responsible for calling {@link #sort()} after all additions are complete.
   *
   * @throws IllegalStateException if the array is full
   */
  public void addUnsorted(int nodeId, float score) {
    if (size >= nodes.length) {
      throw new IllegalStateException(
          "NeighborArray is full (capacity=" + nodes.length + "); cannot add unsorted entry");
    }
    nodes[size] = nodeId;
    scores[size] = score;
    size++;
  }

  /** Sorts entries in descending score order. Call after using {@link #addUnsorted}. */
  public void sort() {
    // Simple insertion sort — optimal for small arrays (R ≤ 128 typical)
    for (int i = 1; i < size; i++) {
      float keyScore = scores[i];
      int keyNode = nodes[i];
      int j = i - 1;
      while (j >= 0 && scores[j] < keyScore) {
        scores[j + 1] = scores[j];
        nodes[j + 1] = nodes[j];
        j--;
      }
      scores[j + 1] = keyScore;
      nodes[j + 1] = keyNode;
    }
  }

  /** Returns true if the given nodeId is present. O(n) linear scan. */
  public boolean contains(int nodeId) {
    for (int i = 0; i < size; i++) {
      if (nodes[i] == nodeId) {
        return true;
      }
    }
    return false;
  }

  /** Clears all entries (logical clear; does not zero arrays). */
  public void clear() {
    size = 0;
  }

  /**
   * Copies all entries from {@code other} into this array, replacing existing content.
   *
   * @throws IllegalArgumentException if {@code other.size()} exceeds this array's capacity
   */
  public void copyFrom(NeighborArray other) {
    if (other.size > nodes.length) {
      throw new IllegalArgumentException(
          "Cannot copy "
              + other.size
              + " entries into array with capacity "
              + nodes.length
              + "; increase maxSize or truncate source first");
    }
    System.arraycopy(other.nodes, 0, this.nodes, 0, other.size);
    System.arraycopy(other.scores, 0, this.scores, 0, other.size);
    this.size = other.size;
  }

  /**
   * Finds the insertion position for a score in descending order. Returns the index where the score
   * should be inserted.
   */
  private int findInsertPosition(float score) {
    int lo = 0;
    int hi = size;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      if (scores[mid] > score) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }
}
