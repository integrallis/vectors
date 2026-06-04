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
import com.integrallis.vectors.ivf.IvfIndex;
import com.integrallis.vectors.ivf.IvfSearchRequest;
import com.integrallis.vectors.ivf.IvfSearchResult;
import java.util.Objects;

/**
 * Read-only {@link IndexSpi} that serves IVF_FLAT search from a <b>pre-decoded</b> {@link IvfIndex}
 * whose vector data comes from an mmap'd {@code vectors.bin}.
 *
 * <p>This is the persistent IVF_FLAT analogue of {@link
 * com.integrallis.vectors.db.index.IvfFlatAdapter} and the open-path terminus: {@code
 * VectorCollectionImpl.openIvfFlatAdapter} reads {@code graph.bin}, extracts the vectors from the
 * mmap'd {@code vectors.bin} into a {@code float[][]}, and calls {@link IvfIndex#decode} to
 * reconstruct the full index without re-running K-Means.
 *
 * <p><b>Build not supported.</b> This adapter is always constructed from a pre-built {@link
 * IvfIndex}; calling {@link #build} throws {@link UnsupportedOperationException}.
 *
 * <p><b>Close semantics.</b> {@link #close()} is a no-op. The mmap'd vectors' lifetime is owned by
 * the per-generation {@link java.lang.foreign.Arena}; {@code VectorCollectionImpl} closes that
 * arena exactly once when its refcount drops to zero. {@link IvfIndex} holds only on-heap state.
 */
public final class MappedIvfFlatAdapter implements IndexSpi {

  private final IvfIndex index;
  private final int nprobe;
  private final float gamma;
  private final int dimension;

  /**
   * Wraps a pre-decoded {@link IvfIndex} together with search parameters.
   *
   * @param index the decoded IVF index (must not be null, must not be mutated after construction)
   * @param nprobe number of clusters probed per query (clamped to [1, k] at search time)
   * @param gamma SOAR spill ratio at search time (0 = disabled)
   * @param dimension the fixed vector dimension (used for query validation)
   * @throws NullPointerException if {@code index} is null
   */
  public MappedIvfFlatAdapter(IvfIndex index, int nprobe, float gamma, int dimension) {
    Objects.requireNonNull(index, "index must not be null");
    this.index = index;
    this.nprobe = nprobe;
    this.gamma = gamma;
    this.dimension = dimension;
  }

  /**
   * Not supported — this adapter is always constructed from a pre-built, on-disk IVF index.
   *
   * @throws UnsupportedOperationException always
   */
  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedIvfFlatAdapter is read-only; use IvfFlatAdapter for in-memory IVF_FLAT or"
            + " construct a MappedIvfFlatAdapter directly from a decoded IvfIndex");
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    if (index.size() == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    // Per-query nprobe override: searchListSize > 0 acts as the IVF beam width for this call,
    // matching the HNSW/Vamana contract; falls back to the constructor-time nprobe otherwise.
    int requestedNprobe = (searchListSize > 0) ? searchListSize : nprobe;
    // Two-pass expansion: when overQueryFactor > 1, probe more clusters (and request more
    // candidates, trimmed to k below), mirroring the in-memory IvfFlatAdapter. Omitting this left
    // over-query inert on the persistent IVF_FLAT path, the same divergence fixed for IVF_PQ
    // (P1.3a).
    boolean twoPass = overQueryFactor > 1.0f;
    int probeCount =
        twoPass
            ? Math.min((int) Math.ceil(requestedNprobe * overQueryFactor), index.k())
            : Math.min(requestedNprobe, index.k());
    int candidateK = twoPass ? (int) Math.ceil(k * overQueryFactor) : k;
    IvfSearchRequest req =
        new IvfSearchRequest(query, candidateK, probeCount, gamma, -Float.MAX_VALUE);
    IvfSearchResult result = index.search(req);
    int sz = Math.min(result.hits().size(), k);
    int[] ordinals = new int[sz];
    float[] scores = new float[sz];
    for (int i = 0; i < sz; i++) {
      ordinals[i] = result.hits().get(i).ordinal();
      scores[i] = result.hits().get(i).score();
    }
    return new SearchOutcome(ordinals, scores);
  }

  @Override
  public int size() {
    return index.size();
  }

  @Override
  public void close() {
    // no-op — IvfIndex is purely on-heap; arena owns the mmap lifetime
  }
}
