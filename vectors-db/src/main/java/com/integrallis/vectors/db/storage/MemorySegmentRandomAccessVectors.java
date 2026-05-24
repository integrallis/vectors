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
package com.integrallis.vectors.db.storage;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Read-only adapter that copies a vector out of a memory-mapped {@link MemorySegmentVectors} into a
 * per-thread scratch {@code float[]} on every {@link #getVector(int)} call. Used by persistent
 * graph-index read paths to bridge mmap'd {@code vectors.bin} into graph-index search APIs.
 *
 * <p>This class simultaneously implements both {@link
 * com.integrallis.vectors.hnsw.RandomAccessVectors} and {@link
 * com.integrallis.vectors.vamana.RandomAccessVectors}. The two interfaces are structurally
 * identical ({@code size()}, {@code dimension()}, {@code getVector(int)}) but live in separate
 * packages to keep each graph-index module dependency-light. Implementing both on one class means a
 * single reader-side bridge is shared by {@link MappedHnswIndexAdapter} and {@link
 * MappedVamanaIndexAdapter}.
 *
 * <p><b>Shared-buffer invariant (CRITICAL).</b> The array returned by {@link #getVector(int)} is a
 * per-thread scratch buffer. Every call on the same thread overwrites the previous call's result.
 * Callers MUST NOT:
 *
 * <ul>
 *   <li>retain the reference across a subsequent {@link #getVector(int)} call on the same thread
 *   <li>compare two scratch-buffer results from consecutive calls (both references will point at
 *       the same buffer holding whichever data the second call wrote)
 *   <li>pass the returned array to code that stashes it in a field or closes over it in a long-
 *       lived lambda
 * </ul>
 *
 * <p>The only safe usage is "fetch, score/compare immediately against a stable reference, discard".
 * This matches both {@code RandomAccessVectors} contracts verbatim ("implementations may return a
 * shared buffer; callers must not retain it across calls") and is the same pattern used by {@code
 * HnswSearcher}'s and {@code VamanaSearcher}'s inner loops.
 *
 * <p><b>Why per-thread, not per-call.</b> Allocating a fresh {@code float[dimension]} on every
 * {@link #getVector(int)} call would issue a young-gen allocation per scored neighbor — at typical
 * M=16, efSearch=100 that's ~100 allocations per search, all escaping due to JIT uncertainty about
 * the array's lifetime. A {@link ThreadLocal} scratch buffer is zero-allocation after the first
 * query per thread and lets the SIMD scoring loops run without GC pressure.
 *
 * <p><b>Why NOT usable by {@link com.integrallis.vectors.hnsw.HnswGraphBuilder} or {@link
 * com.integrallis.vectors.vamana.VamanaGraphBuilder}.</b> Both builders have {@code insert}/{@code
 * link} paths that capture one {@code getVector(x)} result into a local variable and then call
 * {@code vectors.getVector(y)} while still reading the first result. Under the shared-scratch
 * contract that's a use-after-overwrite bug. Only the <i>search</i> paths ({@code
 * HnswSearcher.beamSearch}, {@code VamanaSearcher.search}, both rescore methods) satisfy the
 * single-call-per-iteration contract. {@link MappedHnswIndexAdapter} and {@link
 * MappedVamanaIndexAdapter} — the only users of this class — are constructed from <i>pre-built</i>
 * graphs and reject {@link com.integrallis.vectors.db.index.IndexSpi#build(float[][],
 * com.integrallis.vectors.core.SimilarityFunction)} with {@link UnsupportedOperationException}, so
 * the unsafe builder path cannot reach this adapter.
 *
 * <p><b>Thread safety.</b> Safe for concurrent reads from any number of threads. Each thread lazily
 * allocates its own scratch buffer on first use; the underlying {@link MemorySegmentVectors} is
 * itself thread-safe (shared arena, no mutable state). There is no cross-thread aliasing of the
 * scratch buffer — {@link ThreadLocal#get()} returns a reference unique to the calling thread.
 */
public final class MemorySegmentRandomAccessVectors
    implements com.integrallis.vectors.hnsw.RandomAccessVectors,
        com.integrallis.vectors.vamana.RandomAccessVectors {

  private final MemorySegmentVectors mapped;
  private final int dimension;
  // One scratch float[] per reader thread. Holds the most recently-read vector copied out of the
  // mmap. Overwritten on every getVector() call; callers MUST NOT retain across calls.
  private final ThreadLocal<float[]> scratch;

  /**
   * Wraps the given {@link MemorySegmentVectors} in a shared-scratch read-only view. The {@code
   * mapped} lifetime is NOT tied to this instance — it is owned by the caller-provided {@link
   * java.lang.foreign.Arena} that was passed to {@link MemorySegmentVectors#open}. Creating a
   * wrapper does NOT allocate any scratch buffer; the first {@link #getVector(int)} call on each
   * thread lazily initializes its thread-local {@code float[]}.
   *
   * @throws NullPointerException if {@code mapped} is null
   */
  public MemorySegmentRandomAccessVectors(MemorySegmentVectors mapped) {
    this.mapped = Objects.requireNonNull(mapped, "mapped must not be null");
    this.dimension = mapped.dimension();
    this.scratch = ThreadLocal.withInitial(() -> new float[this.dimension]);
  }

  @Override
  public int size() {
    return mapped.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Returns a per-thread scratch buffer populated with the vector at the given ordinal.
   *
   * <p><b>Shared-buffer:</b> every call on the same thread OVERWRITES the previous return value.
   * See class javadoc for the full invariant.
   *
   * @param ordinal the 0-based vector index
   * @return a per-thread {@code float[dimension]} holding the vector data — do NOT retain across
   *     subsequent calls on this thread
   * @throws IndexOutOfBoundsException if {@code ordinal} is out of range
   */
  @Override
  public float[] getVector(int ordinal) {
    float[] dst = scratch.get();
    MemorySegment slice = mapped.vectorSlice(ordinal);
    MemorySegment.copy(slice, ValueLayout.JAVA_FLOAT, 0L, dst, 0, dimension);
    return dst;
  }

  /**
   * Explicit override to resolve the inherited default from both {@code hnsw.RandomAccessVectors}
   * and {@code vamana.RandomAccessVectors}. Always {@code true} — this adapter returns a per-thread
   * scratch buffer that is overwritten on every call.
   */
  @Override
  public boolean sharesReturnBuffer() {
    return true;
  }
}
