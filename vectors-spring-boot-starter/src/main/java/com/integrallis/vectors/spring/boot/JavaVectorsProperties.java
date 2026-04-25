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
package com.integrallis.vectors.spring.boot;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.VectorCollectionBuilder;
import java.nio.file.Path;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * Spring Boot configuration properties for java-vectors collections.
 *
 * <p>All settings map 1-to-1 to {@link VectorCollectionBuilder} parameters. Example:
 *
 * <pre>{@code
 * java-vectors:
 *   dimension: 1536
 *   metric: COSINE
 *   index-type: HNSW
 *   quantizer: SQ8
 *   storage-path: /var/lib/vectors/my-collection
 *   hnsw:
 *     m: 16
 *     ef-construction: 200
 * }</pre>
 */
@ConfigurationProperties(prefix = "java-vectors")
public class JavaVectorsProperties {

  // -------------------------------------------------------------------------
  // Required
  // -------------------------------------------------------------------------

  /** Vector dimension. <strong>Required.</strong> Must be positive. */
  private int dimension;

  /** Similarity function used for distance computation. <strong>Required.</strong> */
  private SimilarityFunction metric;

  // -------------------------------------------------------------------------
  // Core optional
  // -------------------------------------------------------------------------

  /** Index backend. Default: {@code FLAT}. */
  private IndexType indexType = IndexType.FLAT;

  /** Quantizer to apply after indexing. Default: {@code NONE}. */
  private QuantizerKind quantizer = QuantizerKind.NONE;

  /**
   * If the staging buffer reaches this many documents after an add/addAll, an implicit commit runs
   * automatically. Default: {@link Integer#MAX_VALUE} (disabled).
   */
  private int autoCommitThreshold = Integer.MAX_VALUE;

  /**
   * Absolute path to the collection root directory. When set, the collection uses mmap-backed
   * persistent storage; when {@code null} the collection is fully in-memory.
   */
  private Path storagePath;

  /**
   * Maximum number of cached query results (LRU). {@code 0} (the default) disables the {@link
   * com.integrallis.vectors.db.cache.QvCache} entirely. When positive, queries with {@code
   * includeVector=false} are eligible for caching; results are keyed by a scalar int8 quantization
   * of the query vector combined with {@code k} and a filter predicate hash.
   */
  private int cacheSize = 0;

  // -------------------------------------------------------------------------
  // HNSW parameters
  // -------------------------------------------------------------------------

  /** HNSW-specific parameters. Only used when {@link #indexType} is {@code HNSW}. */
  @NestedConfigurationProperty private HnswProperties hnsw = new HnswProperties();

  // -------------------------------------------------------------------------
  // Vamana parameters
  // -------------------------------------------------------------------------

  /** Vamana-specific parameters. Only used when {@link #indexType} is {@code VAMANA}. */
  @NestedConfigurationProperty private VamanaProperties vamana = new VamanaProperties();

  // -------------------------------------------------------------------------
  // IVF parameters
  // -------------------------------------------------------------------------

  /** IVF-specific parameters. Only used when {@link #indexType} is {@code IVF_FLAT}. */
  @NestedConfigurationProperty private IvfProperties ivf = new IvfProperties();

  // -------------------------------------------------------------------------
  // Quantizer parameters
  // -------------------------------------------------------------------------

  /** PQ/OPQ-specific parameters. Only used when {@link #quantizer} is {@code PQ}. */
  @NestedConfigurationProperty private PqProperties pq = new PqProperties();

  // =========================================================================
  // Nested properties classes
  // =========================================================================

  /** HNSW graph construction parameters. */
  public static class HnswProperties {
    /** Max connections per node (M). Default: {@value VectorCollectionBuilder#DEFAULT_HNSW_M}. */
    private int m = VectorCollectionBuilder.DEFAULT_HNSW_M;

    /**
     * Beam width during graph construction (efConstruction). Default: {@value
     * VectorCollectionBuilder#DEFAULT_HNSW_EF_CONSTRUCTION}.
     */
    private int efConstruction = VectorCollectionBuilder.DEFAULT_HNSW_EF_CONSTRUCTION;

    public int getM() {
      return m;
    }

    public void setM(int m) {
      this.m = m;
    }

    public int getEfConstruction() {
      return efConstruction;
    }

    public void setEfConstruction(int efConstruction) {
      this.efConstruction = efConstruction;
    }
  }

  /** Vamana graph construction parameters. */
  public static class VamanaProperties {
    /**
     * Max out-degree after robust pruning (R). Default: {@value
     * VectorCollectionBuilder#DEFAULT_VAMANA_R}.
     */
    private int maxDegree = VectorCollectionBuilder.DEFAULT_VAMANA_R;

    /**
     * Search list size during construction (L). Default: {@value
     * VectorCollectionBuilder#DEFAULT_VAMANA_L}.
     */
    private int searchListSize = VectorCollectionBuilder.DEFAULT_VAMANA_L;

    /** Pruning factor alpha. Default: {@value VectorCollectionBuilder#DEFAULT_VAMANA_ALPHA}. */
    private float alpha = VectorCollectionBuilder.DEFAULT_VAMANA_ALPHA;

    public int getMaxDegree() {
      return maxDegree;
    }

    public void setMaxDegree(int maxDegree) {
      this.maxDegree = maxDegree;
    }

    public int getSearchListSize() {
      return searchListSize;
    }

    public void setSearchListSize(int searchListSize) {
      this.searchListSize = searchListSize;
    }

    public float getAlpha() {
      return alpha;
    }

    public void setAlpha(float alpha) {
      this.alpha = alpha;
    }
  }

  /** IVF index parameters. */
  public static class IvfProperties {
    /** Number of IVF clusters (K). Default: {@value VectorCollectionBuilder#DEFAULT_IVF_K}. */
    private int k = VectorCollectionBuilder.DEFAULT_IVF_K;

    /**
     * Number of clusters to probe at query time. Default: {@value
     * VectorCollectionBuilder#DEFAULT_IVF_NPROBE}.
     */
    private int nprobe = VectorCollectionBuilder.DEFAULT_IVF_NPROBE;

    /** Max KMeans iterations. Default: {@value VectorCollectionBuilder#DEFAULT_IVF_MAX_ITER}. */
    private int maxIter = VectorCollectionBuilder.DEFAULT_IVF_MAX_ITER;

    public int getK() {
      return k;
    }

    public void setK(int k) {
      this.k = k;
    }

    public int getNprobe() {
      return nprobe;
    }

    public void setNprobe(int nprobe) {
      this.nprobe = nprobe;
    }

    public int getMaxIter() {
      return maxIter;
    }

    public void setMaxIter(int maxIter) {
      this.maxIter = maxIter;
    }
  }

  /** PQ quantizer parameters. */
  public static class PqProperties {
    /** Number of PQ sub-spaces. {@code null} means use the builder default. */
    private Integer subspaces;

    /**
     * Number of clusters per sub-space. Default: {@value
     * VectorCollectionBuilder#DEFAULT_PQ_CLUSTERS}.
     */
    private int clusters = VectorCollectionBuilder.DEFAULT_PQ_CLUSTERS;

    public Integer getSubspaces() {
      return subspaces;
    }

    public void setSubspaces(Integer subspaces) {
      this.subspaces = subspaces;
    }

    public int getClusters() {
      return clusters;
    }

    public void setClusters(int clusters) {
      this.clusters = clusters;
    }
  }

  // =========================================================================
  // Top-level getters and setters
  // =========================================================================

  public int getDimension() {
    return dimension;
  }

  public void setDimension(int dimension) {
    this.dimension = dimension;
  }

  public SimilarityFunction getMetric() {
    return metric;
  }

  public void setMetric(SimilarityFunction metric) {
    this.metric = metric;
  }

  public IndexType getIndexType() {
    return indexType;
  }

  public void setIndexType(IndexType indexType) {
    this.indexType = indexType;
  }

  public QuantizerKind getQuantizer() {
    return quantizer;
  }

  public void setQuantizer(QuantizerKind quantizer) {
    this.quantizer = quantizer;
  }

  public int getAutoCommitThreshold() {
    return autoCommitThreshold;
  }

  public void setAutoCommitThreshold(int autoCommitThreshold) {
    this.autoCommitThreshold = autoCommitThreshold;
  }

  public Path getStoragePath() {
    return storagePath;
  }

  public void setStoragePath(Path storagePath) {
    this.storagePath = storagePath;
  }

  public int getCacheSize() {
    return cacheSize;
  }

  public void setCacheSize(int cacheSize) {
    this.cacheSize = cacheSize;
  }

  public HnswProperties getHnsw() {
    return hnsw;
  }

  public void setHnsw(HnswProperties hnsw) {
    this.hnsw = hnsw;
  }

  public VamanaProperties getVamana() {
    return vamana;
  }

  public void setVamana(VamanaProperties vamana) {
    this.vamana = vamana;
  }

  public IvfProperties getIvf() {
    return ivf;
  }

  public void setIvf(IvfProperties ivf) {
    this.ivf = ivf;
  }

  public PqProperties getPq() {
    return pq;
  }

  public void setPq(PqProperties pq) {
    this.pq = pq;
  }
}
