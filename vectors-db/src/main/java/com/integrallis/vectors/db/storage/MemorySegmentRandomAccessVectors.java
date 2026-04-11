package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.hnsw.RandomAccessVectors;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Read-only {@link RandomAccessVectors} adapter that copies a vector out of a memory-mapped {@link
 * MemorySegmentVectors} into a per-thread scratch {@code float[]} on every {@link #getVector(int)}
 * call. Used exclusively by the persistent HNSW read path (Step 4b) to bridge the mmap'd {@code
 * vectors.bin} into the {@code vectors-hnsw} search API.
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
 * This matches the {@link RandomAccessVectors} contract verbatim ("implementations may return a
 * shared buffer; callers must not retain it across calls") and is the same pattern used by {@code
 * NeighborSelector} and {@code HnswSearcher}'s inner loops.
 *
 * <p><b>Why per-thread, not per-call.</b> Allocating a fresh {@code float[dimension]} on every
 * {@link #getVector(int)} call would issue a young-gen allocation per scored neighbor — at typical
 * M=16, efSearch=100 that's ~100 allocations per search, all escaping due to JIT uncertainty about
 * the array's lifetime. A {@link ThreadLocal} scratch buffer is zero-allocation after the first
 * query per thread and lets the SIMD scoring loops run without GC pressure.
 *
 * <p><b>Why NOT usable by {@link com.integrallis.vectors.hnsw.HnswGraphBuilder}.</b> The builder's
 * {@code insert()} path captures {@code queryVec = getVector(nodeId)} into a local variable and
 * then calls {@code vectors.getVector(neighborId)} while still reading {@code queryVec}. Under the
 * shared-scratch contract that's a use-after-overwrite bug. Phase 5 of Step 4b audited every {@code
 * getVector(} call site in {@code vectors-hnsw/src/main/java} and confirmed that only the
 * <i>search</i> path ({@code HnswSearcher.beamSearch}, {@code greedyDescend}, {@code rescore})
 * satisfies the single-call-per-iteration contract. {@code MappedHnswIndexAdapter}, the sole user
 * of this class, is constructed from a <i>pre-built</i> graph and rejects {@link
 * com.integrallis.vectors.db.index.IndexSpi#build(float[][],
 * com.integrallis.vectors.core.SimilarityFunction)} with {@link UnsupportedOperationException}, so
 * the unsafe builder path cannot reach this adapter.
 *
 * <p><b>Thread safety.</b> Safe for concurrent reads from any number of threads. Each thread lazily
 * allocates its own scratch buffer on first use; the underlying {@link MemorySegmentVectors} is
 * itself thread-safe (shared arena, no mutable state). There is no cross-thread aliasing of the
 * scratch buffer — {@link ThreadLocal#get()} returns a reference unique to the calling thread.
 */
public final class MemorySegmentRandomAccessVectors implements RandomAccessVectors {

  private final MemorySegmentVectors mapped;
  private final int dimension;
  // One scratch float[] per reader thread. Holds the most recently-read vector copied out of the
  // mmap. Overwritten on every getVector() call; callers MUST NOT retain across calls.
  private final ThreadLocal<float[]> scratch;

  /**
   * Wraps the given {@link MemorySegmentVectors} in a {@link RandomAccessVectors} view. The {@code
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
}
