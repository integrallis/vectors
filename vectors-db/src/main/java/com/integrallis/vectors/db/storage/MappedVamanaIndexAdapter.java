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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.vamana.RandomAccessVectors;
import com.integrallis.vectors.vamana.SearchResult;
import com.integrallis.vectors.vamana.VamanaGraph;
import com.integrallis.vectors.vamana.VamanaIndex;
import java.util.Objects;

/**
 * Read-only {@link IndexSpi} that serves Vamana search from a <b>pre-built</b> {@link VamanaGraph}
 * wrapped around a {@link RandomAccessVectors} view of an mmap'd {@code vectors.bin}. This is the
 * persistent-Vamana analogue of {@link com.integrallis.vectors.db.index.VamanaIndexAdapter}: {@code
 * VectorCollectionImpl.openGeneration} decodes {@code graph.bin} via {@link
 * VamanaGraphCodec#decode(byte[])}, wraps the per-generation {@link MemorySegmentVectors} in a
 * {@link MemorySegmentRandomAccessVectors}, and hands both to this adapter's constructor.
 *
 * <p><b>Construction.</b> Unlike {@link com.integrallis.vectors.db.index.VamanaIndexAdapter}, this
 * adapter does <b>not</b> support {@link IndexSpi#build(float[][], SimilarityFunction)} — its data
 * source is an already-built graph from a committed generation. Construct one via {@link
 * #MappedVamanaIndexAdapter(VamanaGraph, RandomAccessVectors, SimilarityFunction)}; calling {@link
 * #build} throws {@link UnsupportedOperationException}. This invariant is load-bearing: {@link
 * MemorySegmentRandomAccessVectors} returns a per-thread scratch buffer from {@link
 * RandomAccessVectors#getVector(int)}, and only the Vamana <i>search</i> path ({@code
 * VamanaSearcher.search}, {@code VamanaSearcher.searchTwoPass}) holds at most one scratch reference
 * per inner iteration and is therefore safe with that contract. The Vamana <i>build</i> path
 * ({@code VamanaGraphBuilder.insert}/{@code link}) holds a query vector across many {@code
 * getVector} calls and would corrupt under shared-scratch — so {@code build} must never reach this
 * adapter.
 *
 * <p><b>Thread safety.</b> Safe for concurrent calls to {@link #search(float[], int, int, float)}
 * from any number of threads. The underlying {@link VamanaIndex} owns a {@link ThreadLocal} pool of
 * {@code VamanaSearcher} instances (one per calling thread), and each searcher's scratch buffers
 * are private. The underlying {@link RandomAccessVectors} is {@link
 * MemorySegmentRandomAccessVectors}, which is also thread-safe via per-thread scratch {@code
 * float[]}.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a <b>no-op</b>. The underlying {@link
 * MemorySegmentVectors}' lifetime is owned by the caller-provided per-generation {@code Arena};
 * {@code VectorCollectionImpl} closes that arena exactly once when its refcount drops to zero.
 * {@link VamanaGraph} is purely on-heap and holds no releasable resources.
 */
public final class MappedVamanaIndexAdapter implements IndexSpi {

  private final VamanaIndex index;
  private final int dimension;

  /**
   * Wraps a pre-built {@link VamanaGraph} together with its backing {@link RandomAccessVectors} and
   * similarity function. None of the arguments may be {@code null}.
   *
   * @param graph the decoded Vamana graph — do NOT mutate after this call, doing so would corrupt
   *     in-flight searches
   * @param vectors read-only random access to the vectors the graph's node IDs refer to; typically
   *     a {@link MemorySegmentRandomAccessVectors} wrapping the generation's mmap'd {@code
   *     vectors.bin}
   * @param metric the similarity function to use for scoring — must match the metric that was used
   *     when the graph was originally built, or query results will be silently wrong
   * @throws NullPointerException if any argument is null
   */
  public MappedVamanaIndexAdapter(
      VamanaGraph graph, RandomAccessVectors vectors, SimilarityFunction metric) {
    Objects.requireNonNull(graph, "graph must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.index = VamanaIndex.ofPrebuilt(graph, vectors, metric);
    this.dimension = vectors.dimension();
  }

  /**
   * Not supported. A {@code MappedVamanaIndexAdapter} is always constructed from a pre-built {@link
   * VamanaGraph} + mmap'd {@link RandomAccessVectors}; there is no "build from an in-memory {@code
   * float[][]}" path here because the data source is an on-disk file produced by the
   * generation-write pipeline.
   *
   * <p><b>This method must remain unsupported</b>. Routing build through this adapter would invoke
   * {@code VamanaGraphBuilder}, whose insert/link paths hold a query vector across multiple {@code
   * getVector} calls on the backing {@link RandomAccessVectors}. When that backing store is a
   * {@link MemorySegmentRandomAccessVectors} (the sole production caller), the per-thread scratch
   * invariant would be violated and the graph would be built from silently corrupt data.
   *
   * @throws UnsupportedOperationException always
   */
  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedVamanaIndexAdapter is read-only; use VamanaIndexAdapter for in-memory Vamana or"
            + " construct a MappedVamanaIndexAdapter directly from a decoded VamanaGraph");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    int size = index.size();
    if (size == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int searchL = Math.max(searchListSize, k);
    SearchResult result;
    if (overQueryFactor > 1.0f && index.isQuantizationEnabled()) {
      result = index.searchTwoPass(query, k, searchL, overQueryFactor);
    } else {
      result = index.search(query, k, searchL);
    }
    return new SearchOutcome(result.nodeIds().clone(), result.scores().clone());
  }

  /**
   * Attaches compressed vectors for quantized two-pass search. Once enabled, {@link
   * #search(float[], int, int, float)} will delegate to {@link VamanaIndex#searchTwoPass} when
   * {@code overQueryFactor > 1.0f}.
   *
   * @param compressed the compressed vectors — must match the index's size and dimension
   * @throws NullPointerException if compressed is null
   */
  public void enableQuantization(CompressedVectors compressed) {
    Objects.requireNonNull(compressed, "compressed must not be null");
    index.enableQuantization(compressed);
  }

  @Override
  public int size() {
    return index.size();
  }

  /**
   * No-op. The underlying {@link MemorySegmentVectors}'s lifetime is tied to its per-generation
   * {@link java.lang.foreign.Arena}; {@code VectorCollectionImpl} closes that arena exactly once
   * per retired generation. The {@link VamanaGraph} and {@link VamanaIndex} are purely on-heap and
   * need no explicit release.
   */
  @Override
  public void close() {
    // no-op — arena owns the mmap lifetime; graph is on-heap
  }
}
