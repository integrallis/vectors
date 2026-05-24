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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntPredicate;

/**
 * Internal index SPI. Implementations wrap a concrete backend (flat scan, HNSW, Vamana, IVF) and
 * expose a uniform search API to the {@link com.integrallis.vectors.db.VectorCollection} facade.
 *
 * <p><b>Filter contract:</b> {@code search()} does <b>not</b> take a filter. The post-filter is
 * applied by the {@code VectorCollection} layer <i>after</i> the SPI returns candidates.
 */
public interface IndexSpi extends AutoCloseable {

  /**
   * Builds the index from scratch from the provided vectors. The vectors are indexed in ordinal
   * order: ordinal {@code i} corresponds to {@code vectors[i]}.
   *
   * @param vectors the vectors to index (dimension must match all rows)
   * @param metric similarity function for the index
   */
  void build(float[][] vectors, SimilarityFunction metric);

  /**
   * Searches the index for the top-{@code k} nearest neighbours.
   *
   * <p><b>Parameter contract.</b> {@code searchListSize} and {@code overQueryFactor} are hints for
   * graph-based and quantized backends that do a coarse pass followed by a rescore. Unquantized
   * brute-force implementations such as {@link FlatScanAdapter} already compare against every
   * stored vector and <i>ignore</i> both parameters — they return identical results regardless of
   * what is passed.
   *
   * @param query the query vector
   * @param k number of final results requested
   * @param searchListSize coarse-pass beam width (may be ignored by brute-force backends)
   * @param overQueryFactor multiplier applied to {@code k} for the coarse pass; expressed as a
   *     {@code float} so non-integer factors like {@code 1.5f} or {@code 2.5f} are supported (may
   *     be ignored by brute-force backends)
   * @return the top-k ordinals and scores (descending)
   */
  SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor);

  /**
   * Multi-start variant of {@link #search}: runs {@code searchMultiStart} independent beam searches
   * from diverse seed nodes in parallel and merges their outputs.
   *
   * <p>Default implementation ignores {@code searchMultiStart} and delegates to the 4-arg {@link
   * #search} — brute-force backends and any backend that does not support multi-start beam search
   * inherit single-start behavior with no behavioural change. Graph-based implementations ({@code
   * HnswIndexAdapter}, {@code MappedHnswIndexAdapter}) override this to route through {@link
   * com.integrallis.vectors.hnsw.HnswIndex#searchMultiStart}.
   *
   * @param query the query vector
   * @param k number of final results requested
   * @param searchListSize coarse-pass beam width (may be ignored by brute-force backends)
   * @param overQueryFactor multiplier applied to {@code k} for the coarse pass
   * @param searchMultiStart number of parallel seed starts; values {@code <= 1} behave identically
   *     to {@link #search}
   * @return the top-k ordinals and scores (descending)
   */
  default SearchOutcome search(
      float[] query, int k, int searchListSize, float overQueryFactor, int searchMultiStart) {
    return search(query, k, searchListSize, overQueryFactor);
  }

  /**
   * ACORN-style pre-filtered search using an ordinal-level predicate.
   *
   * <p>Default implementation ignores the predicate and delegates to {@link #search}; the caller is
   * responsible for applying post-filter. Graph-based implementations ({@code HnswIndexAdapter},
   * {@code MappedHnswIndexAdapter}) override this to route through {@link
   * com.integrallis.vectors.hnsw.HnswIndex#searchFiltered}, enabling navigation through
   * non-matching nodes while only collecting matching results.
   *
   * @param query the query vector
   * @param k number of final results requested
   * @param searchListSize coarse-pass beam width
   * @param overQueryFactor multiplier for coarse-pass k
   * @param predicate ordinal-level filter; {@code true} means the ordinal is eligible for results
   * @return the top-k results that pass the predicate (descending score)
   */
  default SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    // Default: fall back to standard search; post-filter is applied by the caller.
    return search(query, k, searchListSize, overQueryFactor);
  }

  /**
   * Batched search variant: processes {@code queries.length} query vectors and returns one {@link
   * SearchOutcome} per query, preserving input order.
   *
   * <p><b>Default behaviour.</b> Dispatches each query through {@link #search(float[], int, int,
   * float)} on a dedicated virtual thread; all dispatches happen concurrently and the method blocks
   * until every query has completed or thrown. Backends that can amortise work across queries
   * ({@link FlatScanAdapter} fuses distance computation across the corpus) override this to produce
   * real per-batch speedups; backends that cannot inherit the fan-out default unchanged.
   *
   * @param queries non-null, non-empty array of query vectors (all rows must share dimension)
   * @param k number of final results per query
   * @param searchListSize coarse-pass beam width (may be ignored by brute-force backends)
   * @param overQueryFactor multiplier for coarse-pass k (may be ignored by brute-force backends)
   * @return array of length {@code queries.length}; {@code out[i]} corresponds to {@code
   *     queries[i]}
   * @throws IllegalArgumentException if {@code queries} is null or empty
   */
  default SearchOutcome[] searchBatch(
      float[][] queries, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(queries, "queries must not be null");
    if (queries.length == 0) {
      throw new IllegalArgumentException("queries must not be empty");
    }
    if (queries.length == 1) {
      return new SearchOutcome[] {search(queries[0], k, searchListSize, overQueryFactor)};
    }
    SearchOutcome[] out = new SearchOutcome[queries.length];
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<SearchOutcome>> futs = new ArrayList<>(queries.length);
      for (float[] q : queries) {
        futs.add(executor.submit(() -> search(q, k, searchListSize, overQueryFactor)));
      }
      for (int i = 0; i < futs.size(); i++) {
        out[i] = futs.get(i).get();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("searchBatch interrupted", e);
    } catch (ExecutionException e) {
      throw new RuntimeException("searchBatch query failed", e.getCause());
    }
    return out;
  }

  /** Returns the number of vectors currently in the index. */
  int size();

  /** Releases any resources held by the SPI. Default: no-op. */
  @Override
  default void close() {}

  /** Internal carrier for the raw search output (ordinals + scores, descending). */
  record SearchOutcome(int[] ordinals, float[] scores) {
    public SearchOutcome {
      if (ordinals.length != scores.length) {
        throw new IllegalArgumentException(
            "ordinals.length ("
                + ordinals.length
                + ") must match scores.length ("
                + scores.length
                + ")");
      }
    }
  }
}
