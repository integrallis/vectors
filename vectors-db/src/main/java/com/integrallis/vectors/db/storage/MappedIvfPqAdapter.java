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
 * Read-only {@link IndexSpi} that serves IVF-PQ search from a pre-decoded {@link IvfIndex} whose
 * full-precision vector data comes from an mmap'd {@code vectors.bin} and whose PQ codebooks +
 * codes were restored from the codec trailer in {@code graph.bin}.
 *
 * <p>This is the persistent IVF-PQ analogue of {@link
 * com.integrallis.vectors.db.index.IvfPqAdapter}: {@code VectorCollectionImpl.openIvfPqAdapter}
 * reads {@code graph.bin} and the mmap'd vector matrix, then calls {@link IvfIndex#decode} to
 * reconstruct the full index.
 *
 * <p><b>Build not supported.</b> This adapter is always constructed from a pre-built {@link
 * IvfIndex}; calling {@link #build} throws {@link UnsupportedOperationException}.
 */
public final class MappedIvfPqAdapter implements IndexSpi {

  private final IvfIndex index;
  private final int nprobe;
  private final float gamma;
  private final int rescoreFactor;
  private final int dimension;

  /**
   * Wraps a pre-decoded IVF-PQ {@link IvfIndex} together with search parameters.
   *
   * @param index the decoded IVF-PQ index (must not be null; must have {@code isQuantized() ==
   *     true})
   * @param nprobe clusters probed per query (clamped to {@code [1, k]} at search time)
   * @param gamma SOAR spill ratio at search time (0 = disabled)
   * @param rescoreFactor wide-heap multiplier for full-precision rescoring (must be {@code >= 1})
   * @param dimension fixed vector dimension (used for query validation)
   */
  public MappedIvfPqAdapter(
      IvfIndex index, int nprobe, float gamma, int rescoreFactor, int dimension) {
    Objects.requireNonNull(index, "index must not be null");
    if (rescoreFactor < 1) {
      throw new IllegalArgumentException("rescoreFactor must be >= 1: " + rescoreFactor);
    }
    this.index = index;
    this.nprobe = nprobe;
    this.gamma = gamma;
    this.rescoreFactor = rescoreFactor;
    this.dimension = dimension;
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    throw new UnsupportedOperationException(
        "MappedIvfPqAdapter is read-only; use IvfPqAdapter for in-memory IVF_PQ or construct a"
            + " MappedIvfPqAdapter directly from a decoded IvfIndex");
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
    int requestedNprobe = (searchListSize > 0) ? searchListSize : nprobe;
    boolean twoPass = overQueryFactor > 1.0f;
    // Over-query must widen BOTH the probe count and the rescore pool, exactly as the in-memory
    // IvfPqAdapter does. Widening only the rescore pool is near-inert because IVF_PQ recall is
    // gated
    // by cluster coverage: if the true neighbours sit in unprobed clusters, no rescore depth
    // recovers them. Scaling only rescore (the previous behaviour) left persistent IVF_PQ recall
    // flat as overQueryFactor rose, diverging from the in-memory path (P1.3a).
    int probeCount =
        twoPass
            ? Math.min((int) Math.ceil(requestedNprobe * overQueryFactor), index.k())
            : Math.min(requestedNprobe, index.k());
    int effectiveRescore =
        twoPass
            ? Math.max(rescoreFactor, (int) Math.ceil(rescoreFactor * overQueryFactor))
            : rescoreFactor;
    IvfSearchRequest req =
        new IvfSearchRequest(query, k, probeCount, gamma, -Float.MAX_VALUE, effectiveRescore);
    IvfSearchResult result = index.search(req);
    int sz = result.hits().size();
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
    // no-op — arena owns the mmap lifetime; IvfIndex is purely on-heap
  }
}
