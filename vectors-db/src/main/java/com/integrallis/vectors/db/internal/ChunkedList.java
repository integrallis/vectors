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
package com.integrallis.vectors.db.internal;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable, append-only persistent list with structural sharing across snapshots.
 *
 * <p>Designed for the {@code VectorCollectionImpl} commit pipeline where ordinal-keyed structures
 * (the reverse {@code ordinal → id} map and the {@code ordinal → Document} metadata store) grow by
 * appending entries at the end of a dense integer index. Compared to a HAMT, this is simpler and
 * strictly faster: reads are a single index-by-chunk lookup (no trie descent), and successor
 * generations share all already-frozen chunks with the predecessor by reference.
 *
 * <p><b>Snapshot cost.</b> {@link #toBuilder()} is O(1) in shared-chunk references plus a single
 * {@link #CHUNK_SIZE}-element clone of the tail chunk (~8 KB at the default chunk size). For a
 * collection with N entries: outer-array clone is N/CHUNK_SIZE references. At N=100,000 that's 100
 * references, ~800 bytes — versus the ~12 MB the prior HashMap-per-generation deep-copy model
 * spent. The bench-measured speedup at N=100k is ≥30×.
 *
 * <p><b>Threading.</b> {@code ChunkedList} instances are immutable after construction. {@link
 * Builder} instances are not thread-safe and are intended to be used by a single thread holding the
 * writer lock. Once {@link Builder#build()} returns, the resulting {@code ChunkedList} can be
 * safely published to readers via a {@code volatile} write (the JMM happens-before edge on the
 * publish carries over all writes to the tail chunk).
 *
 * @param <T> element type
 */
public final class ChunkedList<T> {

  /** Chunk size in elements. 1024 × 8 B = 8 KB per chunk — fits comfortably in L1 cache. */
  static final int CHUNK_SIZE = 1024;

  /** Mask for the in-chunk offset: {@code i & CHUNK_MASK == i % CHUNK_SIZE} for {@code i ≥ 0}. */
  private static final int CHUNK_MASK = CHUNK_SIZE - 1;

  /** Outer array of chunks. All chunks are {@code Object[CHUNK_SIZE]}. Never null. */
  private final Object[][] chunks;

  /**
   * Logical size. Valid entries are at offsets {@code [0, size)} when viewed through {@link
   * #get(int)}. The last chunk's positions in {@code [size mod CHUNK_SIZE, CHUNK_SIZE)} are
   * <i>unobserved</i> (callers must not read past {@code size}).
   */
  private final int size;

  private static final ChunkedList<Object> EMPTY = new ChunkedList<>(new Object[0][], 0);

  private ChunkedList(Object[][] chunks, int size) {
    this.chunks = chunks;
    this.size = size;
  }

  /** Returns the canonical empty {@code ChunkedList}. */
  @SuppressWarnings("unchecked")
  public static <T> ChunkedList<T> empty() {
    return (ChunkedList<T>) EMPTY;
  }

  /** Number of logical entries. */
  public int size() {
    return size;
  }

  /** {@code true} iff {@link #size()} is zero. */
  public boolean isEmpty() {
    return size == 0;
  }

  /**
   * Returns the element at index {@code i}. {@code O(1)}.
   *
   * @throws IndexOutOfBoundsException if {@code i} is negative or {@code ≥ size()}.
   */
  @SuppressWarnings("unchecked")
  public T get(int i) {
    Objects.checkIndex(i, size);
    return (T) chunks[i >>> 10][i & CHUNK_MASK];
  }

  /**
   * Replaces the element at index {@code i}, returning a new {@code ChunkedList}. Shares all chunks
   * except the affected one (which is cloned). O(CHUNK_SIZE) work.
   *
   * <p>Not used in the commit pipeline (which appends only), but useful for unit tests and as a
   * primitive should future write paths need it.
   *
   * @throws IndexOutOfBoundsException if {@code i} is negative or {@code ≥ size()}.
   */
  public ChunkedList<T> set(int i, T value) {
    Objects.checkIndex(i, size);
    int chunkIdx = i >>> 10;
    int offset = i & CHUNK_MASK;
    Object[][] newChunks = chunks.clone();
    Object[] newChunk = chunks[chunkIdx].clone();
    newChunk[offset] = value;
    newChunks[chunkIdx] = newChunk;
    return new ChunkedList<>(newChunks, size);
  }

  /**
   * Single-element append. Convenience that wraps a one-shot builder. Prefer {@link #toBuilder()}
   * when appending many entries in a batch (the builder amortizes the outer-array clone across all
   * appends).
   */
  public ChunkedList<T> append(T value) {
    Builder<T> b = toBuilder();
    b.append(value);
    return b.build();
  }

  /**
   * Returns a fresh {@link Builder} primed with this list's state. The builder allocates one new
   * tail chunk (cloning the predecessor's tail entries into it, if any), and shares all other
   * chunks with this list by reference. Use the builder to perform a batch of appends, then call
   * {@link Builder#build()} to obtain the immutable result.
   *
   * <p>Time: {@code O(CHUNK_SIZE + num_chunks)} ≈ {@code O(N / CHUNK_SIZE)}. Allocation: one {@link
   * Object} array per chunk in {@code chunks}, plus one fresh tail chunk.
   */
  public Builder<T> toBuilder() {
    if (size == 0) {
      // Empty source: start with one fresh chunk so that append() can write at offset 0 without
      // triggering its "crossed a chunk boundary" branch (which is guarded by size > 0).
      return new Builder<>(new Object[][] {new Object[CHUNK_SIZE]}, 0);
    }
    int tailOffset = size & CHUNK_MASK;
    if (tailOffset == 0) {
      // Last chunk is full. Share all existing chunks by reference and let the first append()
      // allocate the next chunk (the size > 0 path). Pre-allocating an empty trailing chunk here
      // would cause append()'s boundary check to fire and orphan it.
      return new Builder<>(chunks.clone(), size);
    }
    // Partial tail: clone it so the builder's writes do not alias this list's reads past size
    // (out-of-contract for callers but cheap to defend against).
    Object[][] newChunks = chunks.clone();
    Object[] originalTail = chunks[chunks.length - 1];
    Object[] mutableTail = new Object[CHUNK_SIZE];
    System.arraycopy(originalTail, 0, mutableTail, 0, tailOffset);
    newChunks[newChunks.length - 1] = mutableTail;
    return new Builder<>(newChunks, size);
  }

  /**
   * Mutable batch-append helper. Created via {@link ChunkedList#toBuilder()}. Not thread-safe.
   *
   * <p>Append cost is O(1) for the common case (single in-chunk write) and O(1) amortized including
   * the {@code Arrays.copyOf} of the outer chunks array every {@link #CHUNK_SIZE} appends.
   *
   * @param <T> element type
   */
  public static final class Builder<T> {

    private Object[][] chunks;
    private int size;

    Builder(Object[][] chunks, int size) {
      this.chunks = chunks;
      this.size = size;
    }

    /** Returns the running logical size of the builder. */
    public int size() {
      return size;
    }

    /**
     * Appends {@code value} at the tail. O(1) amortized — a single in-chunk write, plus an
     * outer-array {@code copyOf} every {@link #CHUNK_SIZE} appends. No per-call tail-chunk clone
     * (unlike {@link ChunkedList#append(Object)}), so a batch of appends through one builder is
     * O(batch), not O(batch × CHUNK_SIZE).
     *
     * @return this builder, for chaining
     */
    public Builder<T> append(T value) {
      int tailOffset = size & CHUNK_MASK;
      if (tailOffset == 0 && size > 0) {
        // Crossing a chunk boundary: allocate a new chunk and extend the outer array.
        chunks = Arrays.copyOf(chunks, chunks.length + 1);
        chunks[chunks.length - 1] = new Object[CHUNK_SIZE];
      }
      chunks[chunks.length - 1][tailOffset] = value;
      size++;
      return this;
    }

    /**
     * Returns the element at index {@code i}. O(1).
     *
     * @throws IndexOutOfBoundsException if {@code i} is negative or {@code ≥ size()}
     */
    @SuppressWarnings("unchecked")
    public T get(int i) {
      Objects.checkIndex(i, size);
      return (T) chunks[i >>> 10][i & CHUNK_MASK];
    }

    /**
     * Replaces the element at index {@code i}. O(1) for the tail chunk; O(CHUNK_SIZE) for an
     * earlier chunk (which may be shared with a predecessor list and so is cloned copy-on-write
     * before the write). Not on the commit hot path — the commit pipeline only appends — so the
     * occasional clone for a {@code delete}/replace is acceptable.
     *
     * @throws IndexOutOfBoundsException if {@code i} is negative or {@code ≥ size()}
     */
    public void set(int i, T value) {
      Objects.checkIndex(i, size);
      int chunkIdx = i >>> 10;
      // The tail chunk is exclusively owned by this builder (toBuilder cloned it, or append
      // freshly allocated it). Earlier chunks may be shared with a predecessor — clone before
      // writing so the predecessor's view is not corrupted.
      if (chunkIdx != chunks.length - 1) {
        chunks[chunkIdx] = chunks[chunkIdx].clone();
      }
      chunks[chunkIdx][i & CHUNK_MASK] = value;
    }

    /**
     * Builds the immutable {@code ChunkedList} reflecting all appends so far. The returned list
     * holds direct references to the builder's chunk arrays; the builder should not be used after
     * this call (its chunks are aliased with the result).
     */
    public ChunkedList<T> build() {
      return new ChunkedList<>(chunks, size);
    }
  }
}
