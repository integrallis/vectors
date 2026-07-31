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
  // Core (sensible defaults; override as needed)
  // -------------------------------------------------------------------------

  /**
   * Vector dimension. Must be positive.
   *
   * <p>Optional when Spring AI is on the classpath: if left unset (or non-positive) the starter
   * infers the dimension from the application's {@code EmbeddingModel} bean via {@code
   * EmbeddingModel.dimensions()}. Set it explicitly to pin a dimension or when no {@code
   * EmbeddingModel} is available.
   */
  private int dimension;

  /** Similarity function used for distance computation. Default: {@code COSINE}. */
  private SimilarityFunction metric = SimilarityFunction.COSINE;

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

  /**
   * When Spring AI is present, whether the auto-configured {@code JavaVectorsVectorStore} commits
   * after each {@code add(...)} so writes are immediately searchable. Default: {@code true}. Set to
   * {@code false} for batch ingestion where the application commits once per batch. Ignored when no
   * Spring AI {@code EmbeddingModel} is on the classpath (no store is created).
   */
  private boolean commitAfterAdd = true;

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

    /** Returns the maximum number of connections per HNSW node. */
    public int getM() {
      return m;
    }

    /**
     * Sets the maximum number of connections per HNSW node.
     *
     * @param m maximum connections per node
     */
    public void setM(int m) {
      this.m = m;
    }

    /** Returns the HNSW construction beam width. */
    public int getEfConstruction() {
      return efConstruction;
    }

    /**
     * Sets the HNSW construction beam width.
     *
     * @param efConstruction construction beam width
     */
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

    /** Returns the maximum out-degree after Vamana robust pruning. */
    public int getMaxDegree() {
      return maxDegree;
    }

    /**
     * Sets the maximum out-degree after Vamana robust pruning.
     *
     * @param maxDegree maximum graph out-degree
     */
    public void setMaxDegree(int maxDegree) {
      this.maxDegree = maxDegree;
    }

    /** Returns the Vamana construction search-list size. */
    public int getSearchListSize() {
      return searchListSize;
    }

    /**
     * Sets the Vamana construction search-list size.
     *
     * @param searchListSize construction search-list size
     */
    public void setSearchListSize(int searchListSize) {
      this.searchListSize = searchListSize;
    }

    /** Returns the Vamana robust-pruning alpha factor. */
    public float getAlpha() {
      return alpha;
    }

    /**
     * Sets the Vamana robust-pruning alpha factor.
     *
     * @param alpha robust-pruning factor
     */
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

    /** Returns the number of IVF clusters. */
    public int getK() {
      return k;
    }

    /**
     * Sets the number of IVF clusters.
     *
     * @param k cluster count
     */
    public void setK(int k) {
      this.k = k;
    }

    /** Returns the number of IVF clusters probed per query. */
    public int getNprobe() {
      return nprobe;
    }

    /**
     * Sets the number of IVF clusters probed per query.
     *
     * @param nprobe clusters to probe
     */
    public void setNprobe(int nprobe) {
      this.nprobe = nprobe;
    }

    /** Returns the maximum number of KMeans training iterations. */
    public int getMaxIter() {
      return maxIter;
    }

    /**
     * Sets the maximum number of KMeans training iterations.
     *
     * @param maxIter maximum training iterations
     */
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

    /** Returns the configured PQ subspace count, or {@code null} to use the builder default. */
    public Integer getSubspaces() {
      return subspaces;
    }

    /**
     * Sets the PQ subspace count.
     *
     * @param subspaces subspace count, or {@code null} to use the builder default
     */
    public void setSubspaces(Integer subspaces) {
      this.subspaces = subspaces;
    }

    /** Returns the number of clusters in each PQ subspace. */
    public int getClusters() {
      return clusters;
    }

    /**
     * Sets the number of clusters in each PQ subspace.
     *
     * @param clusters clusters per subspace
     */
    public void setClusters(int clusters) {
      this.clusters = clusters;
    }
  }

  // =========================================================================
  // Top-level getters and setters
  // =========================================================================

  /**
   * Returns the configured vector dimension, or a non-positive value when it should be inferred.
   */
  public int getDimension() {
    return dimension;
  }

  /**
   * Sets the vector dimension.
   *
   * @param dimension positive dimension, or a non-positive value to infer it from Spring AI
   */
  public void setDimension(int dimension) {
    this.dimension = dimension;
  }

  /** Returns the similarity function used by the collection. */
  public SimilarityFunction getMetric() {
    return metric;
  }

  /**
   * Sets the similarity function used by the collection.
   *
   * @param metric similarity function
   */
  public void setMetric(SimilarityFunction metric) {
    this.metric = metric;
  }

  /** Returns the configured index backend. */
  public IndexType getIndexType() {
    return indexType;
  }

  /**
   * Sets the index backend.
   *
   * @param indexType index backend
   */
  public void setIndexType(IndexType indexType) {
    this.indexType = indexType;
  }

  /** Returns the configured vector quantizer. */
  public QuantizerKind getQuantizer() {
    return quantizer;
  }

  /**
   * Sets the vector quantizer.
   *
   * @param quantizer quantizer kind
   */
  public void setQuantizer(QuantizerKind quantizer) {
    this.quantizer = quantizer;
  }

  /** Returns the staged-document count that triggers an automatic commit. */
  public int getAutoCommitThreshold() {
    return autoCommitThreshold;
  }

  /**
   * Sets the staged-document count that triggers an automatic commit.
   *
   * @param autoCommitThreshold automatic commit threshold
   */
  public void setAutoCommitThreshold(int autoCommitThreshold) {
    this.autoCommitThreshold = autoCommitThreshold;
  }

  /** Returns the persistent collection path, or {@code null} for an in-memory collection. */
  public Path getStoragePath() {
    return storagePath;
  }

  /**
   * Sets the persistent collection path.
   *
   * @param storagePath collection path, or {@code null} for in-memory storage
   */
  public void setStoragePath(Path storagePath) {
    this.storagePath = storagePath;
  }

  /** Returns the maximum number of cached query results. */
  public int getCacheSize() {
    return cacheSize;
  }

  /**
   * Sets the maximum number of cached query results.
   *
   * @param cacheSize cache capacity; zero disables the cache
   */
  public void setCacheSize(int cacheSize) {
    this.cacheSize = cacheSize;
  }

  /** Returns whether the Spring AI vector store commits after every add operation. */
  public boolean isCommitAfterAdd() {
    return commitAfterAdd;
  }

  /**
   * Sets whether the Spring AI vector store commits after every add operation.
   *
   * @param commitAfterAdd {@code true} to make each add immediately searchable
   */
  public void setCommitAfterAdd(boolean commitAfterAdd) {
    this.commitAfterAdd = commitAfterAdd;
  }

  /** Returns the HNSW-specific properties. */
  public HnswProperties getHnsw() {
    return hnsw;
  }

  /**
   * Sets the HNSW-specific properties.
   *
   * @param hnsw HNSW configuration
   */
  public void setHnsw(HnswProperties hnsw) {
    this.hnsw = hnsw;
  }

  /** Returns the Vamana-specific properties. */
  public VamanaProperties getVamana() {
    return vamana;
  }

  /**
   * Sets the Vamana-specific properties.
   *
   * @param vamana Vamana configuration
   */
  public void setVamana(VamanaProperties vamana) {
    this.vamana = vamana;
  }

  /** Returns the IVF-specific properties. */
  public IvfProperties getIvf() {
    return ivf;
  }

  /**
   * Sets the IVF-specific properties.
   *
   * @param ivf IVF configuration
   */
  public void setIvf(IvfProperties ivf) {
    this.ivf = ivf;
  }

  /** Returns the PQ-specific properties. */
  public PqProperties getPq() {
    return pq;
  }

  /**
   * Sets the PQ-specific properties.
   *
   * @param pq PQ configuration
   */
  public void setPq(PqProperties pq) {
    this.pq = pq;
  }
}
