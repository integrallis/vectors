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
import com.integrallis.vectors.vamana.VamanaIndex;
import com.integrallis.vectors.vamana.VamanaTopology;
import java.util.Objects;

/**
 * Disk-resident {@link IndexSpi} that serves Vamana search from a {@link PagedVamanaTopology}
 * (graph adjacency paged from the mmap'd {@code graph.bin}) plus a {@link RandomAccessVectors} view
 * of the mmap'd {@code vectors.bin} (I.4). It is the sole persistent Vamana adapter: identical
 * search semantics and thread-safety to in-memory Vamana, but the graph topology is never decoded
 * into heap — search heap is O(threads·L), not O(N·R).
 *
 * <p>Construction-only (no {@link #build}); the data source is an already-committed generation.
 * Only the search path is reached here, never the build path — which matters because {@link
 * MemorySegmentRandomAccessVectors} returns a shared per-thread scratch buffer that the build path
 * (unlike search) would corrupt by holding across multiple {@code getVector} calls.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a no-op — both the graph segment and the vector
 * segment are owned by the caller-provided per-generation {@code Arena}, which {@code
 * VectorCollectionImpl} closes exactly once at refcount zero.
 */
public final class MappedVamanaPagedIndexAdapter implements IndexSpi {

  private final VamanaIndex index;
  private final int dimension;

  /**
   * Wraps a paged {@link VamanaTopology} together with its backing {@link RandomAccessVectors} and
   * similarity function.
   *
   * @param topology the paged graph view over the mmap'd {@code graph.bin}
   * @param vectors read-only random access to the vectors (typically a {@link
   *     MemorySegmentRandomAccessVectors} over the mmap'd {@code vectors.bin})
   * @param metric similarity function — must match the metric the graph was built with
   * @throws NullPointerException if any argument is null
   */
  public MappedVamanaPagedIndexAdapter(
      VamanaTopology topology, RandomAccessVectors vectors, SimilarityFunction metric) {
    Objects.requireNonNull(topology, "topology must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.index = VamanaIndex.ofPrebuilt(topology, vectors, metric);
    this.dimension = vectors.dimension();
  }

  /**
   * Not supported — this adapter serves an already-committed generation. Routing build through a
   * mmap-backed adapter is unsafe: {@code VamanaGraphBuilder}'s insert/link paths hold a query
   * vector across subsequent {@code getVector} calls, violating the shared-scratch contract of
   * {@link MemorySegmentRandomAccessVectors}.
   *
   * @throws UnsupportedOperationException always
   */
  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedVamanaPagedIndexAdapter is read-only; use VamanaIndexAdapter for in-memory Vamana");
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
   * Attaches compressed vectors for quantized two-pass search (delegates to {@link
   * VamanaIndex#searchTwoPass} when {@code overQueryFactor > 1.0f}). The coarse traversal still
   * uses the paged topology; only the scorer changes.
   *
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

  @Override
  public void close() {
    // no-op — the per-generation arena owns both the graph and vector mmap lifetimes
  }
}
