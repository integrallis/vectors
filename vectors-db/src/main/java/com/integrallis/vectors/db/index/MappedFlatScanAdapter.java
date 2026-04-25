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
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.db.storage.MemorySegmentVectors;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Persistent brute-force {@link IndexSpi} that scores every stored vector against the query using
 * <b>zero-copy SIMD loads</b> directly from a memory-mapped {@code vectors.bin}. This is the
 * mmap-backed analogue of {@link FlatScanAdapter} and the headline Step 4a read path:
 *
 * <pre>
 *   mmap'd page → MemorySegment → FloatVector.fromMemorySegment → vmovups/ld1 → FMA
 * </pre>
 *
 * No intermediate {@code float[]} copy, no {@code ByteBuffer}, no scalar load — the mmap'd page
 * <i>is</i> the SIMD register input.
 *
 * <p><b>Construction.</b> Unlike {@link FlatScanAdapter}, this adapter does <b>not</b> support the
 * {@link IndexSpi#build(float[][], SimilarityFunction)} path — its data source is an already-built
 * {@link MemorySegmentVectors} that came out of the Step 4a generation-write pipeline. Construct
 * one via {@link #MappedFlatScanAdapter(MemorySegmentVectors, SimilarityFunction)}; calling {@link
 * #build(float[][], SimilarityFunction)} throws {@link UnsupportedOperationException}.
 *
 * <p><b>Scoring.</b> Results are bit-level compatible with {@link FlatScanAdapter} in the sense
 * that the same {@link SimilarityFunction} normalizations are applied to the raw distance/dot
 * products (EUCLIDEAN → {@code 1/(1+sqDist)}, DOT_PRODUCT → {@code (1+dot)/2}, COSINE → {@code
 * (1+cos)/2}, MAXIMUM_INNER_PRODUCT → {@link SimilarityFunction#scaleMaxInnerProductScore(float)}).
 * The raw SIMD primitives on the array path ({@code PanamaVectorUtilSupport.dotProduct(float[],
 * float[])}) use 4x FMA unrolling, while the MemorySegment path ({@code
 * PanamaVectorUtilSupport.dotProduct(MemorySegment, MemorySegment, int)}) uses 2x unrolling, so
 * reduction order can differ by a handful of ULPs on FMA-capable hardware. Top-k ordering is stable
 * for reasonable datasets; score values match to within {@code ~1e-4f}.
 *
 * <p><b>Ignored parameters.</b> Both {@code searchListSize} and {@code overQueryFactor} are ignored
 * for the same reason as {@link FlatScanAdapter}: brute force already examines every vector, so the
 * parameters cannot affect the output. See {@link IndexSpi#search(float[], int, int, float)}.
 *
 * <p><b>Per-call allocation.</b> Each {@link #search(float[], int, int, float)} call allocates
 * exactly one off-heap {@link MemorySegment} of size {@code dimension * 4} bytes to hold the query,
 * backed by a {@link Arena#ofConfined()}. The arena is closed at method return via
 * try-with-resources. All subsequent distance evaluations reuse that single segment against slices
 * of the mmap'd file — zero extra off-heap allocations per candidate.
 *
 * <p><b>Thread safety.</b> Concurrent calls to {@link #search(float[], int, int, float)} from
 * multiple threads are safe: the underlying {@link MemorySegmentVectors} is thread-safe (its whole
 * point — {@code Arena.ofShared()}), and each search allocates its own confined arena for the query
 * buffer. {@link #build(float[][], SimilarityFunction)} is not supported, so there is no mutation
 * path to synchronize.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a <b>no-op</b>. The underlying {@link
 * MemorySegmentVectors}'s lifetime is owned by the caller-provided {@code Arena} passed to {@link
 * MemorySegmentVectors#open(java.nio.file.Path, int, int, Arena)} (typically the per-generation
 * arena), and calling close() here must not touch that arena. {@code VectorCollectionImpl} is
 * responsible for closing the shared arena exactly once per retired generation.
 */
public final class MappedFlatScanAdapter implements IndexSpi {

  private final MemorySegmentVectors store;
  private final SimilarityFunction metric;
  private final int dimension;
  private final MemorySegment segment;
  private final int rawVectorByteSize;

  /**
   * Wraps an already-opened {@link MemorySegmentVectors} for brute-force search under the given
   * similarity function. Neither parameter may be {@code null}.
   *
   * @param store the mapped vector store; its lifetime is NOT tied to this adapter
   * @param metric the similarity function to apply to raw distance/dot products
   * @throws NullPointerException if either argument is null
   */
  public MappedFlatScanAdapter(MemorySegmentVectors store, SimilarityFunction metric) {
    this.store = Objects.requireNonNull(store, "store must not be null");
    this.metric = Objects.requireNonNull(metric, "metric must not be null");
    this.dimension = store.dimension();
    this.segment = store.segment();
    this.rawVectorByteSize = store.rawVectorByteSize();
  }

  /**
   * Not supported. A {@code MappedFlatScanAdapter} is always constructed from a pre-built {@link
   * MemorySegmentVectors}; there is no "build from an in-memory {@code float[][]}" path here
   * because the data source is an on-disk file produced by the generation-write pipeline.
   *
   * @throws UnsupportedOperationException always
   */
  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedFlatScanAdapter is constructed from a pre-built MemorySegmentVectors; "
            + "use the constructor instead of build()");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int size = store.size();
    if (size == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }

    int actualK = Math.min(k, size);

    // Bounded min-heap (by score) over at most actualK entries. When full, the root is the
    // worst-so-far kept result; a new candidate with strictly higher score replaces the root.
    int[] heapIds = new int[actualK];
    float[] heapScores = new float[actualK];
    int heapSize = 0;

    // Single per-call off-heap allocation: the query segment. Confined arena is closed on exit.
    try (Arena queryArena = Arena.ofConfined()) {
      MemorySegment queryBuf = queryArena.allocate(ValueLayout.JAVA_FLOAT, dimension);
      MemorySegment.copy(query, 0, queryBuf, ValueLayout.JAVA_FLOAT, 0, dimension);

      for (int ord = 0; ord < size; ord++) {
        long off = store.vectorOffsetFor(ord);
        MemorySegment vec = segment.asSlice(off, rawVectorByteSize);
        // Switch expression (not statement) so the compiler enforces exhaustiveness over
        // SimilarityFunction at build time — adding a new enum constant without a new case
        // here becomes a compile error, not a silent runtime IllegalStateException.
        float score =
            switch (metric) {
              case EUCLIDEAN -> 1f / (1f + VectorUtil.squareDistance(queryBuf, vec, dimension));
              case DOT_PRODUCT -> (1f + VectorUtil.dotProduct(queryBuf, vec, dimension)) / 2f;
              case COSINE -> (1f + VectorUtil.cosine(queryBuf, vec, dimension)) / 2f;
              case MAXIMUM_INNER_PRODUCT ->
                  SimilarityFunction.scaleMaxInnerProductScore(
                      VectorUtil.dotProduct(queryBuf, vec, dimension));
            };

        if (heapSize < actualK) {
          heapIds[heapSize] = ord;
          heapScores[heapSize] = score;
          heapSize++;
          siftUp(heapIds, heapScores, heapSize - 1);
        } else if (score > heapScores[0]) {
          heapIds[0] = ord;
          heapScores[0] = score;
          siftDown(heapIds, heapScores, 0, heapSize);
        }
      }
    }

    // Drain heap into a descending-sorted result array.
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

  @Override
  public int size() {
    return store.size();
  }

  /**
   * No-op. The underlying {@link MemorySegmentVectors}'s lifetime is tied to its caller-provided
   * {@link Arena}; {@code VectorCollectionImpl} closes that arena exactly once per retired
   * generation.
   */
  @Override
  public void close() {
    // no-op — arena owns the mmap lifetime
  }

  /** Sifts the element at {@code idx} up the min-heap to restore the heap invariant. */
  private static void siftUp(int[] ids, float[] scores, int idx) {
    while (idx > 0) {
      int parent = (idx - 1) >>> 1;
      if (scores[parent] <= scores[idx]) {
        break;
      }
      float tmpScore = scores[parent];
      int tmpId = ids[parent];
      scores[parent] = scores[idx];
      ids[parent] = ids[idx];
      scores[idx] = tmpScore;
      ids[idx] = tmpId;
      idx = parent;
    }
  }

  /** Sifts the element at {@code idx} down the min-heap to restore the heap invariant. */
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
      float tmpScore = scores[smallest];
      int tmpId = ids[smallest];
      scores[smallest] = scores[idx];
      ids[smallest] = ids[idx];
      scores[idx] = tmpScore;
      ids[idx] = tmpId;
      idx = smallest;
    }
  }
}
