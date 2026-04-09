package com.integrallis.vectors.hnsw;

/**
 * Sorted parallel arrays of (nodeId, score) pairs, used for neighbor lists in HNSW graphs and as
 * scratch space during search.
 *
 * <p>Entries are maintained in <b>descending score order</b> (best first, worst at {@code size-1}).
 * This enables O(1) worst-neighbor access for eviction and ordered iteration for the diversity
 * heuristic.
 *
 * <p>For typical HNSW parameters (M=16-32), linear scan and {@link System#arraycopy} are faster
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
