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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.cache.QvCache;
import java.nio.file.Path;

/**
 * Fluent builder for {@link VectorCollection}.
 *
 * <p>Required settings: {@link #dimension(int)} and {@link #metric(SimilarityFunction)}. {@link
 * #build()} throws {@link IllegalStateException} if either is unset.
 *
 * <p>Supports {@link IndexType#FLAT}, {@link IndexType#HNSW}, {@link IndexType#VAMANA}, {@link
 * IndexType#IVF_FLAT}, and {@link IndexType#IVF_PQ} with supported quantizers in either in-memory
 * mode (no {@link #storagePath(Path)}) or persistent mmap-backed mode (absolute {@code
 * storagePath}).
 */
public final class VectorCollectionBuilder {

  /** HNSW {@code M} parameter default. Matches the HnswIndex.Builder default. */
  public static final int DEFAULT_HNSW_M = 16;

  /** HNSW {@code efConstruction} parameter default. Matches the HnswIndex.Builder default. */
  public static final int DEFAULT_HNSW_EF_CONSTRUCTION = 200;

  /**
   * HNSW build-time thread count default. Defaults to {@code 1} (deterministic, single-threaded) so
   * the graph encoding is bit-exact reproducible; call {@link #hnswBuildThreads(int)} to opt in to
   * parallel construction via {@link com.integrallis.vectors.hnsw.ConcurrentHnswGraphBuilder}.
   */
  public static final int DEFAULT_HNSW_BUILD_THREADS = 1;

  /**
   * Vamana build-time thread count default. Defaults to {@code 1} (deterministic, single-threaded)
   * so the graph encoding is bit-exact reproducible; call {@link #vamanaBuildThreads(int)} to opt
   * in to parallel construction via {@code ConcurrentVamanaGraphBuilder}.
   */
  public static final int DEFAULT_VAMANA_BUILD_THREADS = 1;

  /** Vamana {@code R} (maxDegree) parameter default. Matches the VamanaIndex.Builder default. */
  public static final int DEFAULT_VAMANA_R = 64;

  /**
   * Vamana {@code L} (searchListSize) parameter default. Matches the VamanaIndex.Builder default.
   */
  public static final int DEFAULT_VAMANA_L = 128;

  /** Vamana {@code alpha} parameter default. Matches the VamanaIndex.Builder default. */
  public static final float DEFAULT_VAMANA_ALPHA = 1.2f;

  /** Default PQ number of clusters per subspace. */
  public static final int DEFAULT_PQ_CLUSTERS = 256;

  /** Default RaBitQ random seed. */
  public static final long DEFAULT_RABIT_SEED = 42L;

  /** Default TurboQuant per-coordinate bit-width. */
  public static final int DEFAULT_TURBO_BITS = 8;

  /** Default TurboQuant rotation seed. */
  public static final long DEFAULT_TURBO_SEED = 42L;

  /**
   * Default TurboQuant mode: the unbiased two-stage TurboQuant_prod (MSE + QJL residual), the
   * paper's recommended variant for nearest-neighbour search.
   */
  public static final boolean DEFAULT_TURBO_UNBIASED = true;

  /** Default IVF number of clusters (K). */
  public static final int DEFAULT_IVF_K = 16;

  /** Default IVF probe count. */
  public static final int DEFAULT_IVF_NPROBE = 4;

  /** Default IVF KMeans max iterations. */
  public static final int DEFAULT_IVF_MAX_ITER = 30;

  /** Default IVF KMeans seed. */
  public static final long DEFAULT_IVF_SEED = 42L;

  private Integer dimension;
  private SimilarityFunction metric;
  private IndexType indexType = IndexType.FLAT;
  private QuantizerKind quantizerKind = QuantizerKind.NONE;
  private int autoCommitThreshold = Integer.MAX_VALUE;
  private Path storageRoot;
  private int hnswM = DEFAULT_HNSW_M;
  private int hnswEfConstruction = DEFAULT_HNSW_EF_CONSTRUCTION;
  private int hnswBuildThreads = DEFAULT_HNSW_BUILD_THREADS;
  private int vamanaMaxDegree = DEFAULT_VAMANA_R;
  private int vamanaSearchListSize = DEFAULT_VAMANA_L;
  private float vamanaAlpha = DEFAULT_VAMANA_ALPHA;
  private Long vamanaSeed; // lazily filled with System.nanoTime() at build() time if unset
  private int vamanaBuildThreads = DEFAULT_VAMANA_BUILD_THREADS;

  // IVF-specific params (shared across IVF_FLAT and IVF_PQ)
  private int ivfK = DEFAULT_IVF_K;
  private int ivfNprobe = DEFAULT_IVF_NPROBE;
  private int ivfMaxIter = DEFAULT_IVF_MAX_ITER;
  private float ivfGamma = 0f;
  private boolean ivfSoar = false;
  private long ivfSeed = DEFAULT_IVF_SEED;

  // IVF_PQ-only params (null = use defaults derived from dimension at build() time)
  private Integer ivfPqSubspaces;
  private Integer ivfPqClusters;
  private Float ivfPqAnisotropicThreshold;
  private int ivfRescoreFactor = 1;

  // cuVS-specific params (null = use defaults derived from indexType at build() time)
  private VectorCollectionConfig.CuVsParams cuvsParams;

  // QvCache — 0 means disabled
  private int cacheSize = 0;

  // Quantizer-specific params (all nullable — null means "use defaults")
  private Integer pqSubspaces;
  private Integer pqClusters;
  private Boolean pqCenter;
  private Integer pqTrainThreads;
  private Boolean bqBbq;
  private Long rabitSeed;
  private Integer nvqSubvectors;
  private Integer turboBits;
  private Long turboSeed;
  private Boolean turboUnbiased;

  VectorCollectionBuilder() {}

  /** Sets the required fixed vector dimension. */
  public VectorCollectionBuilder dimension(int dimension) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    return this;
  }

  /** Sets the required similarity function. */
  public VectorCollectionBuilder metric(SimilarityFunction metric) {
    if (metric == null) {
      throw new IllegalArgumentException("metric must not be null");
    }
    this.metric = metric;
    return this;
  }

  /** Selects the index backend. */
  public VectorCollectionBuilder indexType(IndexType indexType) {
    if (indexType == null) {
      throw new IllegalArgumentException("indexType must not be null");
    }
    this.indexType = indexType;
    return this;
  }

  /** Selects the quantizer. */
  public VectorCollectionBuilder quantizer(QuantizerKind quantizerKind) {
    if (quantizerKind == null) {
      throw new IllegalArgumentException("quantizerKind must not be null");
    }
    this.quantizerKind = quantizerKind;
    return this;
  }

  /**
   * Sets the HNSW {@code M} parameter (max connections per upper layer). Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#HNSW}. Must be positive. Default: {@value
   * #DEFAULT_HNSW_M}.
   */
  public VectorCollectionBuilder hnswM(int m) {
    if (m <= 0) {
      throw new IllegalArgumentException("M must be positive: " + m);
    }
    this.hnswM = m;
    return this;
  }

  /**
   * Sets the HNSW {@code efConstruction} parameter (beam width during graph construction). Ignored
   * unless {@link #indexType(IndexType)} is {@link IndexType#HNSW}. Must be {@code >= hnswM(...)}.
   * Default: {@value #DEFAULT_HNSW_EF_CONSTRUCTION}.
   *
   * <p><b>Persistence note.</b> {@code efConstruction} is NOT stored on disk — only the structural
   * parameters captured in {@code graph.bin} (specifically {@code M}) survive a close/reopen cycle.
   * A reopened collection that triggers a fresh commit will use whichever {@code efConstruction}
   * the caller sets on the next {@code VectorCollection.builder()} invocation, or the default if
   * unset.
   */
  public VectorCollectionBuilder hnswEfConstruction(int efConstruction) {
    if (efConstruction <= 0) {
      throw new IllegalArgumentException("efConstruction must be positive: " + efConstruction);
    }
    this.hnswEfConstruction = efConstruction;
    return this;
  }

  /**
   * Sets the number of worker threads used during HNSW graph construction. Values {@code > 1} route
   * the build through {@link com.integrallis.vectors.hnsw.ConcurrentHnswGraphBuilder}, which
   * produces valid (but non-deterministic) graphs with equivalent recall. Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#HNSW}. Must be {@code >= 1}. Default: {@link
   * #DEFAULT_HNSW_BUILD_THREADS} (half of {@code Runtime.availableProcessors()}).
   */
  public VectorCollectionBuilder hnswBuildThreads(int threads) {
    if (threads < 1) {
      throw new IllegalArgumentException("threads must be >= 1: " + threads);
    }
    this.hnswBuildThreads = threads;
    return this;
  }

  /**
   * Sets the Vamana {@code R} (maxDegree) parameter — max out-degree after robust pruning. Ignored
   * unless {@link #indexType(IndexType)} is {@link IndexType#VAMANA}. Must be positive. Default:
   * {@value #DEFAULT_VAMANA_R}.
   */
  public VectorCollectionBuilder vamanaMaxDegree(int maxDegree) {
    if (maxDegree <= 0) {
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    }
    this.vamanaMaxDegree = maxDegree;
    return this;
  }

  /**
   * Sets the Vamana {@code L} (searchListSize) parameter — beam width during graph construction.
   * Ignored unless {@link #indexType(IndexType)} is {@link IndexType#VAMANA}. Must be {@code >=
   * vamanaMaxDegree(...)}. Default: {@value #DEFAULT_VAMANA_L}.
   *
   * <p><b>Persistence note.</b> {@code searchListSize} is NOT stored on disk — only the structural
   * parameter {@code maxDegree} captured in {@code graph.bin} survives a close/reopen cycle. A
   * reopened collection that triggers a fresh commit will use whichever {@code searchListSize} the
   * caller sets on the next {@code VectorCollection.builder()} invocation, or the default if unset.
   */
  public VectorCollectionBuilder vamanaSearchListSize(int searchListSize) {
    if (searchListSize <= 0) {
      throw new IllegalArgumentException("searchListSize must be positive: " + searchListSize);
    }
    this.vamanaSearchListSize = searchListSize;
    return this;
  }

  /**
   * Sets the Vamana {@code alpha} diversity parameter used by the robust pruner. Ignored unless
   * {@link #indexType(IndexType)} is {@link IndexType#VAMANA}. Must be {@code >= 1.0}. Default:
   * {@value #DEFAULT_VAMANA_ALPHA}.
   *
   * <p><b>Persistence note.</b> {@code alpha} is NOT stored on disk. A reopened collection that
   * triggers a fresh commit will use whichever value the caller sets on the next {@code
   * VectorCollection.builder()} invocation.
   */
  public VectorCollectionBuilder vamanaAlpha(float alpha) {
    if (alpha < 1.0f) {
      throw new IllegalArgumentException("alpha must be >= 1.0: " + alpha);
    }
    this.vamanaAlpha = alpha;
    return this;
  }

  /**
   * Sets the random seed used by {@code VamanaGraphBuilder} for deterministic graph construction.
   * Ignored unless {@link #indexType(IndexType)} is {@link IndexType#VAMANA}. Default: {@code
   * System.nanoTime()} at {@link #build()} time — explicit seeds produce byte-identical graphs
   * across runs with the same data, which is essential for regression testing.
   */
  public VectorCollectionBuilder vamanaSeed(long seed) {
    this.vamanaSeed = seed;
    return this;
  }

  /**
   * Sets the number of worker threads used by {@code ConcurrentVamanaGraphBuilder} during
   * construction. Ignored unless {@link #indexType(IndexType)} is {@link IndexType#VAMANA}. Must be
   * {@code >= 1}. Default: {@link #DEFAULT_VAMANA_BUILD_THREADS} (single-threaded, deterministic).
   * Values {@code > 1} trade determinism for wall-clock parallelism; recall is preserved within
   * statistical noise.
   */
  public VectorCollectionBuilder vamanaBuildThreads(int threads) {
    if (threads < 1) {
      throw new IllegalArgumentException("threads must be >= 1: " + threads);
    }
    this.vamanaBuildThreads = threads;
    return this;
  }

  // ---------------------------------------------------------------------------
  // IVF-specific parameter setters
  // ---------------------------------------------------------------------------

  /**
   * Sets the IVF number of clusters (K). Ignored unless {@link #indexType(IndexType)} is {@link
   * IndexType#IVF_FLAT}. Must be positive. Default: {@value #DEFAULT_IVF_K}.
   */
  public VectorCollectionBuilder ivfK(int k) {
    if (k <= 0) throw new IllegalArgumentException("IVF k must be positive: " + k);
    this.ivfK = k;
    return this;
  }

  /**
   * Sets the IVF probe count — the number of clusters searched per query. Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#IVF_FLAT}. Must be in [1, k]. Default: {@value
   * #DEFAULT_IVF_NPROBE}.
   */
  public VectorCollectionBuilder ivfNprobe(int nprobe) {
    if (nprobe <= 0) throw new IllegalArgumentException("IVF nprobe must be positive: " + nprobe);
    this.ivfNprobe = nprobe;
    return this;
  }

  /**
   * Sets the maximum KMeans iterations for IVF cluster training. Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#IVF_FLAT}. Must be positive. Default: {@value
   * #DEFAULT_IVF_MAX_ITER}.
   */
  public VectorCollectionBuilder ivfMaxIter(int maxIter) {
    if (maxIter <= 0)
      throw new IllegalArgumentException("IVF maxIter must be positive: " + maxIter);
    this.ivfMaxIter = maxIter;
    return this;
  }

  /**
   * Sets the IVF SOAR spill ratio gamma in [0, 1]. Ignored unless {@link #indexType(IndexType)} is
   * {@link IndexType#IVF_FLAT}. Default: 0 (no spill).
   */
  public VectorCollectionBuilder ivfGamma(float gamma) {
    if (gamma < 0f || gamma > 1f)
      throw new IllegalArgumentException("IVF gamma must be in [0, 1]: " + gamma);
    this.ivfGamma = gamma;
    return this;
  }

  /**
   * Enables SOAR-style cluster spill during IVF search. Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#IVF_FLAT}. Default: false.
   */
  public VectorCollectionBuilder ivfSoar(boolean soar) {
    this.ivfSoar = soar;
    return this;
  }

  /**
   * Sets the RNG seed for IVF KMeans++ initialisation. Ignored unless {@link #indexType(IndexType)}
   * is {@link IndexType#IVF_FLAT} or {@link IndexType#IVF_PQ}. Default: {@value #DEFAULT_IVF_SEED}.
   */
  public VectorCollectionBuilder ivfSeed(long seed) {
    this.ivfSeed = seed;
    return this;
  }

  /**
   * Sets the IVF-PQ number of subspaces (M). Ignored unless {@link #indexType(IndexType)} is {@link
   * IndexType#IVF_PQ}. Must be positive; the chosen M should divide the vector dimension. Default:
   * {@code max(1, dimension / 8)}.
   */
  public VectorCollectionBuilder ivfPqSubspaces(int numSubspaces) {
    if (numSubspaces <= 0) {
      throw new IllegalArgumentException("ivfPqSubspaces must be positive: " + numSubspaces);
    }
    this.ivfPqSubspaces = numSubspaces;
    return this;
  }

  /**
   * Sets the IVF-PQ centroids per subspace (Ks). Ignored unless {@link #indexType(IndexType)} is
   * {@link IndexType#IVF_PQ}. Must be in {@code [2, 256]}. Default: {@value #DEFAULT_PQ_CLUSTERS}.
   */
  public VectorCollectionBuilder ivfPqClusters(int numClusters) {
    if (numClusters < 2 || numClusters > 256) {
      throw new IllegalArgumentException("ivfPqClusters must be in [2, 256]: " + numClusters);
    }
    this.ivfPqClusters = numClusters;
    return this;
  }

  /**
   * Sets the IVF-PQ anisotropic threshold used by the coordinate-descent encoder. Pass {@code -1f}
   * for standard (unweighted) PQ, or a value in {@code [0, 1]} to enable anisotropic encoding.
   * Ignored unless {@link #indexType(IndexType)} is {@link IndexType#IVF_PQ}. Default: {@code -1f}.
   */
  public VectorCollectionBuilder ivfPqAnisotropicThreshold(float threshold) {
    if (threshold != -1f && (threshold < 0f || threshold > 1f)) {
      throw new IllegalArgumentException(
          "ivfPqAnisotropicThreshold must be -1 or in [0, 1]: " + threshold);
    }
    this.ivfPqAnisotropicThreshold = threshold;
    return this;
  }

  /**
   * Sets the IVF-PQ search-time rescore factor — the wide heap sized {@code k × rescoreFactor} is
   * re-ranked against full-precision vectors before the top-k is returned. Must be {@code >= 1};
   * {@code 1} disables rescoring (raw ADC scores are returned). Ignored unless {@link
   * #indexType(IndexType)} is {@link IndexType#IVF_PQ}. Default: {@code 1}.
   */
  public VectorCollectionBuilder ivfRescoreFactor(int rescoreFactor) {
    if (rescoreFactor < 1) {
      throw new IllegalArgumentException("ivfRescoreFactor must be >= 1: " + rescoreFactor);
    }
    this.ivfRescoreFactor = rescoreFactor;
    return this;
  }

  /**
   * Sets cuVS GPU index parameters. Ignored unless {@link #indexType(IndexType)} is {@link
   * IndexType#CUVS_BRUTEFORCE} or {@link IndexType#CUVS_CAGRA}. When unset, defaults are derived
   * from the chosen {@link IndexType} at {@link #build()} time.
   */
  public VectorCollectionBuilder cuvsParams(VectorCollectionConfig.CuVsParams params) {
    this.cuvsParams = params;
    return this;
  }

  // ---------------------------------------------------------------------------
  // Quantizer-specific parameter setters
  // ---------------------------------------------------------------------------

  /**
   * Sets the PQ number of subspaces (M). Only used when {@link #quantizer(QuantizerKind)} is {@link
   * QuantizerKind#PQ}. Must be positive and must evenly divide the vector dimension. Default:
   * {@code max(1, dimension / 8)}.
   */
  public VectorCollectionBuilder pqSubspaces(int numSubspaces) {
    if (numSubspaces <= 0) {
      throw new IllegalArgumentException("numSubspaces must be positive: " + numSubspaces);
    }
    this.pqSubspaces = numSubspaces;
    return this;
  }

  /**
   * Sets the PQ number of clusters per subspace (Ks). Only used when {@link
   * #quantizer(QuantizerKind)} is {@link QuantizerKind#PQ}. Must be in [2, 256]. Default: {@value
   * #DEFAULT_PQ_CLUSTERS}.
   */
  public VectorCollectionBuilder pqClusters(int numClusters) {
    if (numClusters < 2 || numClusters > 256) {
      throw new IllegalArgumentException("numClusters must be in [2, 256]: " + numClusters);
    }
    this.pqClusters = numClusters;
    return this;
  }

  /**
   * Sets whether PQ subtracts a global centroid before quantization. Only used when {@link
   * #quantizer(QuantizerKind)} is {@link QuantizerKind#PQ}. Default: true.
   */
  public VectorCollectionBuilder pqCenter(boolean center) {
    this.pqCenter = center;
    return this;
  }

  /**
   * Sets the number of worker threads used by per-subspace k-means during PQ training. Ignored
   * unless {@link #quantizer(QuantizerKind)} is {@link QuantizerKind#PQ}. Must be {@code >= 1}.
   * Default: {@code 1} (single-threaded, byte-identical to pre-R2.E releases). Values {@code > 1}
   * route through the parallel train path of {@link
   * com.integrallis.vectors.quantization.ProductQuantizer}; the resulting codebook is deterministic
   * but numerically distinct from the sequential codebook.
   */
  public VectorCollectionBuilder pqTrainThreads(int threads) {
    if (threads < 1) {
      throw new IllegalArgumentException("threads must be >= 1: " + threads);
    }
    this.pqTrainThreads = threads;
    return this;
  }

  /**
   * Sets the BQ mode. Only used when {@link #quantizer(QuantizerKind)} is {@link QuantizerKind#BQ}.
   * Pass {@code true} for BBQ (Better Binary Quantization — computes a centroid and per-vector
   * corrections for asymmetric distance estimation), or {@code false} for plain sign-bit mode.
   * Default: true (BBQ is strictly more accurate).
   */
  public VectorCollectionBuilder bqMode(boolean bbq) {
    this.bqBbq = bbq;
    return this;
  }

  /**
   * Sets the random seed for RaBitQ's rotation matrix. Only used when {@link
   * #quantizer(QuantizerKind)} is {@link QuantizerKind#RABITQ}. Default: {@value
   * #DEFAULT_RABIT_SEED}.
   */
  public VectorCollectionBuilder rabitSeed(long seed) {
    this.rabitSeed = seed;
    return this;
  }

  /**
   * Sets the NVQ number of subvectors (M). Only used when {@link #quantizer(QuantizerKind)} is
   * {@link QuantizerKind#NVQ}. Must be positive and must evenly divide the vector dimension.
   * Default: {@code max(1, dimension / 4)}.
   */
  public VectorCollectionBuilder nvqSubvectors(int numSubvectors) {
    if (numSubvectors <= 0) {
      throw new IllegalArgumentException("numSubvectors must be positive: " + numSubvectors);
    }
    this.nvqSubvectors = numSubvectors;
    return this;
  }

  /**
   * Sets the TurboQuant per-coordinate bit-width. Only used when {@link #quantizer(QuantizerKind)}
   * is {@link QuantizerKind#TURBOQUANT}. Must be in {@code [1, 8]}. Default: {@value
   * #DEFAULT_TURBO_BITS}.
   */
  public VectorCollectionBuilder turboBits(int bits) {
    if (bits < 1 || bits > 8) {
      throw new IllegalArgumentException("TurboQuant bits must be in [1, 8]: " + bits);
    }
    this.turboBits = bits;
    return this;
  }

  /**
   * Sets the random seed for TurboQuant's rotation. Only used when {@link
   * #quantizer(QuantizerKind)} is {@link QuantizerKind#TURBOQUANT}. Default: {@value
   * #DEFAULT_TURBO_SEED}.
   */
  public VectorCollectionBuilder turboSeed(long seed) {
    this.turboSeed = seed;
    return this;
  }

  /**
   * Selects the TurboQuant variant. {@code true} (default) uses the unbiased two-stage
   * TurboQuant_prod (MSE + QJL residual) recommended by the paper for nearest-neighbour search;
   * {@code false} uses the smaller/faster MSE-only variant (biased inner products). Only used when
   * {@link #quantizer(QuantizerKind)} is {@link QuantizerKind#TURBOQUANT}.
   */
  public VectorCollectionBuilder turboUnbiased(boolean unbiased) {
    this.turboUnbiased = unbiased;
    return this;
  }

  /**
   * Sets the staging buffer size at which {@code add}/{@code addAll} auto-commit before returning.
   * Must be positive. Pass {@link Integer#MAX_VALUE} to disable auto-commit (the default), which
   * forces the caller to drive {@link VectorCollection#commit()} explicitly.
   */
  public VectorCollectionBuilder autoCommitThreshold(int autoCommitThreshold) {
    if (autoCommitThreshold <= 0) {
      throw new IllegalArgumentException(
          "autoCommitThreshold must be positive (use Integer.MAX_VALUE to disable): "
              + autoCommitThreshold);
    }
    this.autoCommitThreshold = autoCommitThreshold;
    return this;
  }

  /**
   * Enables persistent mmap-backed mode rooted at {@code storageRoot}. The directory is created if
   * it does not already exist. On {@link #build()}, the collection runs the crash-recovery sweep
   * via {@link com.integrallis.vectors.db.storage.GenerationDirectory#recover} and opens the
   * resulting generation through a shared {@link java.lang.foreign.Arena}.
   *
   * <p>Passing {@code null} explicitly disables persistence (the default). The {@code storageRoot}
   * must be an absolute path when non-null; the builder stores whatever the caller passed, so the
   * caller owns resolving relative paths upfront.
   */
  public VectorCollectionBuilder storagePath(Path storageRoot) {
    this.storageRoot = storageRoot;
    return this;
  }

  /**
   * Enables the {@link QvCache} query-result cache with the given LRU capacity.
   *
   * <p>Cached results are keyed by a scalar int8 quantization of the query vector combined with
   * {@code k} and a hash of the filter predicate. The cache is automatically invalidated after
   * every {@link VectorCollection#commit()}.
   *
   * @param maxEntries maximum number of cached query results; {@code 0} (the default) disables the
   *     cache entirely
   */
  public VectorCollectionBuilder cacheSize(int maxEntries) {
    if (maxEntries < 0) throw new IllegalArgumentException("cacheSize must be >= 0: " + maxEntries);
    this.cacheSize = maxEntries;
    return this;
  }

  /** Builds the collection. */
  public VectorCollection build() {
    if (dimension == null) {
      throw new IllegalStateException("dimension is required, call builder.dimension(d)");
    }
    if (metric == null) {
      throw new IllegalStateException("metric is required, call builder.metric(m)");
    }
    if (storageRoot != null && !storageRoot.isAbsolute()) {
      throw new IllegalArgumentException(
          "storagePath must be absolute when non-null (the collection must not depend on the JVM"
              + " working directory): "
              + storageRoot);
    }
    VectorCollectionConfig.HnswParams hnswParams =
        (indexType == IndexType.HNSW)
            ? new VectorCollectionConfig.HnswParams(hnswM, hnswEfConstruction, hnswBuildThreads)
            : null;
    VectorCollectionConfig.VamanaParams vamanaParams =
        (indexType == IndexType.VAMANA)
            ? new VectorCollectionConfig.VamanaParams(
                vamanaMaxDegree,
                vamanaSearchListSize,
                vamanaAlpha,
                vamanaSeed != null ? vamanaSeed : System.nanoTime(),
                vamanaBuildThreads)
            : null;
    VectorCollectionConfig.IvfParams ivfParams =
        (indexType == IndexType.IVF_FLAT)
            ? new VectorCollectionConfig.IvfParams(
                ivfK, Math.min(ivfNprobe, ivfK), ivfMaxIter, ivfGamma, ivfSoar, ivfSeed)
            : null;
    VectorCollectionConfig.IvfPqParams ivfPqParams =
        (indexType == IndexType.IVF_PQ)
            ? new VectorCollectionConfig.IvfPqParams(
                ivfK,
                Math.min(ivfNprobe, ivfK),
                ivfMaxIter,
                ivfGamma,
                ivfSoar,
                ivfSeed,
                ivfPqSubspaces != null ? ivfPqSubspaces : Math.max(1, dimension / 8),
                ivfPqClusters != null ? ivfPqClusters : DEFAULT_PQ_CLUSTERS,
                ivfPqAnisotropicThreshold != null ? ivfPqAnisotropicThreshold : -1f,
                ivfRescoreFactor)
            : null;
    VectorCollectionConfig.CuVsParams effectiveCuvsParams =
        switch (indexType) {
          case CUVS_BRUTEFORCE ->
              cuvsParams != null
                  ? cuvsParams
                  : VectorCollectionConfig.CuVsParams.BruteForce.defaults();
          case CUVS_CAGRA ->
              cuvsParams != null ? cuvsParams : VectorCollectionConfig.CuVsParams.Cagra.defaults();
          case FLAT, HNSW, VAMANA, IVF_FLAT, IVF_PQ -> null;
        };
    QuantizerParams quantizerParams = buildQuantizerParams();
    var config =
        new VectorCollectionConfig(
            dimension,
            metric,
            indexType,
            quantizerKind,
            autoCommitThreshold,
            storageRoot,
            hnswParams,
            vamanaParams,
            quantizerParams,
            ivfParams,
            effectiveCuvsParams,
            ivfPqParams);
    QvCache cache = cacheSize > 0 ? new QvCache(cacheSize) : QvCache.DISABLED;
    return new VectorCollectionImpl(config, cache);
  }

  /**
   * Builds the {@link QuantizerParams} record appropriate for the current {@link #quantizerKind},
   * applying defaults for any unset parameters. Returns {@code null} for {@link
   * QuantizerKind#NONE}.
   */
  private QuantizerParams buildQuantizerParams() {
    return switch (quantizerKind) {
      case NONE -> null;
      case SQ8, SQ4 -> new QuantizerParams.ScalarParams();
      case PQ ->
          new QuantizerParams.PqParams(
              pqSubspaces != null ? pqSubspaces : Math.max(1, dimension / 8),
              pqClusters != null ? pqClusters : DEFAULT_PQ_CLUSTERS,
              pqCenter != null ? pqCenter : true,
              pqTrainThreads != null ? pqTrainThreads : 1);
      case BQ -> new QuantizerParams.BqParams(bqBbq != null ? bqBbq : true);
      case RABITQ ->
          new QuantizerParams.RaBitParams(rabitSeed != null ? rabitSeed : DEFAULT_RABIT_SEED);
      case NVQ ->
          new QuantizerParams.NvqParams(
              nvqSubvectors != null ? nvqSubvectors : Math.max(1, dimension / 4));
      case TURBOQUANT ->
          new QuantizerParams.TurboParams(
              turboBits != null ? turboBits : DEFAULT_TURBO_BITS,
              turboSeed != null ? turboSeed : DEFAULT_TURBO_SEED,
              turboUnbiased != null ? turboUnbiased : DEFAULT_TURBO_UNBIASED);
    };
  }
}
