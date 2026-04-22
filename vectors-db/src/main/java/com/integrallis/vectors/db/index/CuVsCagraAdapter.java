package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.VectorCollectionConfig;
import com.integrallis.vectors.gpu.GpuAvailability;
import com.integrallis.vectors.gpu.GpuProvider;
import com.integrallis.vectors.gpu.GpuUnavailableException;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.Objects;

/**
 * {@link IndexSpi} backed by the NVIDIA cuVS CAGRA GPU graph index. Build and search require a
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
public final class CuVsCagraAdapter implements IndexSpi {

  private final VectorCollectionConfig.CuVsParams.Cagra params;
  private final Object lock = new Object();

  private CuVSResources resources;
  private CagraIndex index;
  private int size;
  private int dimension;
  private SimilarityFunction metric;

  public CuVsCagraAdapter(VectorCollectionConfig.CuVsParams.Cagra params) {
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
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withGraphDegree(params.graphDegree())
              .withIntermediateGraphDegree(params.intermediateGraphDegree())
              .withMetric(toCuvsDistance(metric))
              .build();
      this.index =
          CagraIndex.newBuilder(resources)
              .withIndexParams(indexParams)
              .withDataset(vectors)
              .build();
    } catch (Throwable t) {
      closeQuietly();
      throw new GpuUnavailableException("cuVS CAGRA build failed: " + t.getMessage(), t);
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
        int effectiveItopk = Math.max(params.itopkSize(), Math.max(searchListSize, k));
        CagraSearchParams searchParams =
            new CagraSearchParams.Builder()
                .withItopkSize(effectiveItopk)
                .withMaxIterations(params.maxIterations())
                .build();
        CuVSMatrix queryMatrix = CuVSMatrix.ofArray(new float[][] {query});
        CagraQuery cagraQuery =
            new CagraQuery.Builder(resources)
                .withSearchParams(searchParams)
                .withQueryVectors(queryMatrix)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .withTopK(Math.min(k, size))
                .build();
        SearchResults results = index.search(cagraQuery);
        return CuVsResultDecoder.decode(results, metric);
      } catch (Throwable t) {
        throw new GpuUnavailableException("cuVS CAGRA search failed: " + t.getMessage(), t);
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

  private static CuvsDistanceType toCuvsDistance(SimilarityFunction metric) {
    return switch (metric) {
      case EUCLIDEAN -> CuvsDistanceType.L2Expanded;
      case COSINE -> CuvsDistanceType.CosineExpanded;
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> CuvsDistanceType.InnerProduct;
    };
  }
}
