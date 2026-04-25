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
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.gpu.GpuAvailability;
import com.integrallis.vectors.gpu.GpuProvider;
import com.integrallis.vectors.gpu.GpuUnavailableException;
import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.Objects;

/**
 * {@link IndexSpi} backed by NVIDIA cuVS brute-force GPU scoring. Build and search require a
 * compatible CUDA device; on hosts where {@link GpuProvider} reports unavailable, {@link
 * #build(float[][], SimilarityFunction)} throws {@link GpuUnavailableException}.
 *
 * <p>In-memory only — persistent mmap mode is not yet supported.
 *
 * <p><b>Throughput ceiling.</b> Concurrent searches are serialized through a single synchronized
 * monitor because cuVS {@link CuVSResources} are documented as non-thread-safe. Effective QPS is
 * therefore bounded by the latency of a single GPU search, regardless of how many caller threads
 * issue queries. Multi-start parallelism ({@code searchMultiStart}) is also a no-op on this
 * backend. To saturate a GPU with this adapter, batch queries at the application layer; native cuVS
 * batched search will ship in a later release. This limitation will be lifted once cuVS-java
 * exposes thread-safe resources or an official batched-query API.
 */
public final class CuVsBruteForceAdapter implements IndexSpi {

  @SuppressWarnings("unused") // Reserved for future tunables (writer threads, metric overrides).
  private final VectorCollectionConfig.CuVsParams.BruteForce params;

  private final Object lock = new Object();

  private CuVSResources resources;
  private BruteForceIndex index;
  private int size;
  private int dimension;
  private SimilarityFunction metric;

  public CuVsBruteForceAdapter(VectorCollectionConfig.CuVsParams.BruteForce params) {
    this.params = Objects.requireNonNull(params, "params must not be null");
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    GpuAvailability a = GpuProvider.availability();
    if (!a.isAvailable()) {
      throw new GpuUnavailableException(a.reason().orElse("no GPU detected"));
    }
    this.size = vectors.length;
    this.dimension = vectors.length == 0 ? 0 : vectors[0].length;
    this.metric = metric;
    if (vectors.length == 0) {
      this.index = null;
      return;
    }
    try {
      closeQuietly();
      this.resources = CuVSProvider.provider().newCuVSResources(CuVSProvider.tempDirectory());
      this.index =
          BruteForceIndex.newBuilder(resources)
              .withIndexParams(new BruteForceIndexParams.Builder().build())
              .withDataset(vectors)
              .build();
    } catch (Throwable t) {
      closeQuietly();
      throw new GpuUnavailableException("cuVS brute-force build failed: " + t.getMessage(), t);
    }
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (size == 0 || index == null) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    synchronized (lock) {
      try {
        BruteForceQuery bfq =
            new BruteForceQuery(
                new float[][] {query},
                SearchResults.IDENTITY_MAPPING,
                Math.min(k, size),
                null,
                size,
                resources);
        SearchResults results = index.search(bfq);
        return CuVsResultDecoder.decode(results, metric);
      } catch (Throwable t) {
        throw new GpuUnavailableException("cuVS brute-force search failed: " + t.getMessage(), t);
      }
    }
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void close() {
    closeQuietly();
  }

  private void closeQuietly() {
    if (index != null) {
      try {
        index.close();
      } catch (Exception ignored) {
        // Best-effort cleanup.
      }
      index = null;
    }
    if (resources != null) {
      try {
        resources.close();
      } catch (Exception ignored) {
        // Best-effort cleanup.
      }
      resources = null;
    }
  }
}
