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
import java.nio.file.Path;
import java.util.Objects;

/**
 * Immutable collection configuration. Captured on {@code build()} and persisted in {@code
 * manifest.bin} for on-disk collections.
 *
 * @param dimension fixed vector dimension (must be positive)
 * @param metric similarity function (never null)
 * @param indexType index backend (never null)
 * @param quantizerKind quantizer selector (never null; use {@link QuantizerKind#NONE} if unused)
 * @param autoCommitThreshold if the staging buffer reaches this many documents after an {@code
 *     add}/{@code addAll}, an implicit {@code commit()} runs before the write path returns. Must be
 *     positive. Use {@link Integer#MAX_VALUE} to disable auto-commit entirely (the default).
 * @param storageRoot absolute collection root for persistent mode. When {@code null} the collection
 *     operates as an in-memory-only collection. When non-null, every {@code commit()} writes a new
 *     generation directory under this path via {@link
 *     com.integrallis.vectors.db.storage.GenerationDirectory} and mmaps the result for the next
 *     read snapshot.
 * @param hnswParams HNSW build-time parameters. Must be {@code non-null iff indexType == HNSW}; any
 *     other combination is rejected by the compact constructor.
 * @param vamanaParams Vamana build-time parameters. Must be {@code non-null iff indexType ==
 *     VAMANA}; any other combination is rejected by the compact constructor.
 * @param quantizerParams build-time quantizer parameters. Must be {@code null} when {@code
 *     quantizerKind == NONE}. When non-NONE, may be {@code null} to use defaults. When non-null,
 *     the record type must match the quantizer kind.
 * @param preserveOriginalVectors when {@code false} (the default) and {@link #metric()} is {@link
 *     SimilarityFunction#COSINE}, vectors are L2-unit-normalized at ingest and the index scores
 *     them with {@link SimilarityFunction#DOT_PRODUCT} — cosine of unit vectors equals their dot
 *     product, so this is rank- and score-identical to cosine while running the cheaper single
 *     reduction kernel (see {@link #indexMetric()}). When {@code true}, vectors are stored verbatim
 *     and scored with the true cosine kernel. Ignored for non-COSINE metrics.
 */
public record VectorCollectionConfig(
    int dimension,
    SimilarityFunction metric,
    IndexType indexType,
    QuantizerKind quantizerKind,
    int autoCommitThreshold,
    Path storageRoot,
    HnswParams hnswParams,
    VamanaParams vamanaParams,
    QuantizerParams quantizerParams,
    IvfParams ivfParams,
    CuVsParams cuvsParams,
    IvfPqParams ivfPqParams,
    boolean preserveOriginalVectors) {

  /**
   * Returns {@code true} when vectors should be L2-unit-normalized at ingest/search so the index
   * can score them with {@link SimilarityFunction#DOT_PRODUCT} instead of the true {@link
   * SimilarityFunction#COSINE} kernel. This is exactly {@code metric == COSINE &&
   * !preserveOriginalVectors}. For unit vectors {@code cosine == dot}, so the substitution is rank-
   * and score-identical while doing one reduction instead of three.
   */
  public boolean normalizeForCosine() {
    return metric == SimilarityFunction.COSINE && !preserveOriginalVectors;
  }

  /**
   * The similarity function the index is actually built and searched with. Returns {@link
   * SimilarityFunction#DOT_PRODUCT} when {@link #normalizeForCosine()} (the stored vectors are
   * unit-length, so dot equals cosine), otherwise the true {@link #metric()}. Public callers that
   * need the collection's declared metric must use {@link #metric()}, which always reports the true
   * metric (e.g. COSINE) regardless of this internal optimization.
   */
  public SimilarityFunction indexMetric() {
    return normalizeForCosine() ? SimilarityFunction.DOT_PRODUCT : metric;
  }

  public VectorCollectionConfig {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    Objects.requireNonNull(metric, "metric must not be null");
    Objects.requireNonNull(indexType, "indexType must not be null");
    Objects.requireNonNull(quantizerKind, "quantizerKind must not be null");
    if (autoCommitThreshold <= 0) {
      throw new IllegalArgumentException(
          "autoCommitThreshold must be positive (use Integer.MAX_VALUE to disable): "
              + autoCommitThreshold);
    }
    // storageRoot is intentionally nullable — null = in-memory mode.
    if ((indexType == IndexType.HNSW) != (hnswParams != null)) {
      throw new IllegalArgumentException(
          "hnswParams must be non-null iff indexType == HNSW (indexType="
              + indexType
              + ", hnswParams="
              + (hnswParams == null ? "null" : "set")
              + ")");
    }
    if ((indexType == IndexType.VAMANA) != (vamanaParams != null)) {
      throw new IllegalArgumentException(
          "vamanaParams must be non-null iff indexType == VAMANA (indexType="
              + indexType
              + ", vamanaParams="
              + (vamanaParams == null ? "null" : "set")
              + ")");
    }
    // quantizerParams must be null when quantizerKind is NONE.
    if (quantizerKind == QuantizerKind.NONE && quantizerParams != null) {
      throw new IllegalArgumentException("quantizerParams must be null when quantizerKind is NONE");
    }
    if ((indexType == IndexType.IVF_FLAT) != (ivfParams != null)) {
      throw new IllegalArgumentException(
          "ivfParams must be non-null iff indexType == IVF_FLAT (indexType="
              + indexType
              + ", ivfParams="
              + (ivfParams == null ? "null" : "set")
              + ")");
    }
    if ((indexType == IndexType.IVF_PQ) != (ivfPqParams != null)) {
      throw new IllegalArgumentException(
          "ivfPqParams must be non-null iff indexType == IVF_PQ (indexType="
              + indexType
              + ", ivfPqParams="
              + (ivfPqParams == null ? "null" : "set")
              + ")");
    }
    boolean isCuvs = indexType == IndexType.CUVS_BRUTEFORCE || indexType == IndexType.CUVS_CAGRA;
    if (isCuvs != (cuvsParams != null)) {
      throw new IllegalArgumentException(
          "cuvsParams must be non-null iff indexType in {CUVS_BRUTEFORCE, CUVS_CAGRA} (indexType="
              + indexType
              + ", cuvsParams="
              + (cuvsParams == null ? "null" : "set")
              + ")");
    }
    if (indexType == IndexType.CUVS_BRUTEFORCE && !(cuvsParams instanceof CuVsParams.BruteForce)) {
      throw new IllegalArgumentException(
          "indexType CUVS_BRUTEFORCE requires CuVsParams.BruteForce, got "
              + cuvsParams.getClass().getSimpleName());
    }
    if (indexType == IndexType.CUVS_CAGRA && !(cuvsParams instanceof CuVsParams.Cagra)) {
      throw new IllegalArgumentException(
          "indexType CUVS_CAGRA requires CuVsParams.Cagra, got "
              + cuvsParams.getClass().getSimpleName());
    }
    if (isCuvs && storageRoot != null) {
      throw new IllegalArgumentException(
          "CUVS_* index types do not support persistent storage yet (storageRoot must be null)");
    }
    if (isCuvs && quantizerKind != QuantizerKind.NONE) {
      throw new IllegalArgumentException(
          "CUVS_* index types do not support quantization (quantizerKind must be NONE, got "
              + quantizerKind
              + ")");
    }
  }

  /**
   * 12-arg convenience constructor that defaults {@link #preserveOriginalVectors()} to {@code
   * false} (the #A cosine-normalization optimization is on by default). Preserves the shape of
   * every call site written before {@code preserveOriginalVectors} was added.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot,
      HnswParams hnswParams,
      VamanaParams vamanaParams,
      QuantizerParams quantizerParams,
      IvfParams ivfParams,
      CuVsParams cuvsParams,
      IvfPqParams ivfPqParams) {
    this(
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
        cuvsParams,
        ivfPqParams,
        false);
  }

  /**
   * 11-arg convenience constructor that defaults {@link #ivfPqParams()} to {@code null}. Preserves
   * the pre-IVF_PQ canonical shape for call sites written before Session 2 of the IVF-PQ effort.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot,
      HnswParams hnswParams,
      VamanaParams vamanaParams,
      QuantizerParams quantizerParams,
      IvfParams ivfParams,
      CuVsParams cuvsParams) {
    this(
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
        cuvsParams,
        null);
  }

  /**
   * 8-arg convenience constructor that defaults {@link #quantizerParams()}, {@link #ivfParams()},
   * {@link #cuvsParams()}, and {@link #ivfPqParams()} to {@code null}.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot,
      HnswParams hnswParams,
      VamanaParams vamanaParams) {
    this(
        dimension,
        metric,
        indexType,
        quantizerKind,
        autoCommitThreshold,
        storageRoot,
        hnswParams,
        vamanaParams,
        null,
        null,
        null,
        null);
  }

  /**
   * 7-arg convenience constructor that defaults {@link #vamanaParams()}, {@link
   * #quantizerParams()}, {@link #ivfParams()}, {@link #cuvsParams()}, and {@link #ivfPqParams()} to
   * {@code null}. Throws via the compact constructor if the caller asks for {@link
   * IndexType#VAMANA} without supplying {@link VamanaParams}.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot,
      HnswParams hnswParams) {
    this(
        dimension,
        metric,
        indexType,
        quantizerKind,
        autoCommitThreshold,
        storageRoot,
        hnswParams,
        null,
        null,
        null,
        null,
        null);
  }

  /**
   * 6-arg convenience constructor that defaults {@link #hnswParams()}, {@link #vamanaParams()}, and
   * {@link #quantizerParams()} to {@code null}. Suitable for flat-scan collections. Throws via the
   * compact constructor if the caller asks for {@link IndexType#HNSW} or {@link IndexType#VAMANA}
   * without supplying the matching parameter record.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot) {
    this(
        dimension,
        metric,
        indexType,
        quantizerKind,
        autoCommitThreshold,
        storageRoot,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  /**
   * 5-arg convenience constructor for in-memory, flat-scan collections. Kept for test fixtures that
   * pre-date the IVF_FLAT, CUVS, and IVF_PQ support.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold) {
    this(
        dimension,
        metric,
        indexType,
        quantizerKind,
        autoCommitThreshold,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  /**
   * HNSW graph-construction parameters captured on {@code VectorCollection.builder().build()}. Only
   * {@code M} is persisted on disk (inside {@code graph.bin}); {@code efConstruction} and {@code
   * threads} are build-time hints that are NOT preserved across a close/reopen, so a reopened
   * collection that triggers another commit will use whichever values the caller sets on the new
   * builder invocation.
   *
   * @param m max connections per upper layer (must be positive)
   * @param efConstruction beam width during construction (must be {@code >= m})
   * @param threads worker threads for graph construction. {@code 0} means auto (parallel across all
   *     cores for large builds, single-threaded for small ones); explicit values must be {@code >=
   *     1}, where {@code 1} forces deterministic single-threaded construction.
   */
  public record HnswParams(int m, int efConstruction, int threads) {
    public HnswParams {
      if (m <= 0) {
        throw new IllegalArgumentException("M must be positive: " + m);
      }
      if (efConstruction < m) {
        throw new IllegalArgumentException(
            "efConstruction (" + efConstruction + ") must be >= M (" + m + ")");
      }
      if (threads < 0) {
        throw new IllegalArgumentException("threads must be >= 0 (0 = auto): " + threads);
      }
    }

    /** Convenience constructor defaulting to single-threaded (deterministic) construction. */
    public HnswParams(int m, int efConstruction) {
      this(m, efConstruction, 1);
    }
  }

  /**
   * Vamana graph-construction parameters captured on {@code VectorCollection.builder().build()}.
   * Only {@code maxDegree} is persisted on disk (inside {@code graph.bin}'s header); {@code
   * searchListSize}, {@code alpha}, and {@code seed} are all build-time hints that are NOT
   * preserved across a close/reopen, so a reopened collection that triggers another commit will use
   * whichever values the caller sets on the new builder invocation.
   *
   * <p>Naming mirrors the Vamana paper: {@code R} = maxDegree, {@code L} = searchListSize, {@code
   * alpha} is the robust-pruner diversity parameter (must be {@code >= 1.0}). {@code seed} drives
   * {@code VamanaGraphBuilder}'s random initialization so consecutive commits with the same data
   * produce byte-identical graphs.
   *
   * @param maxDegree Vamana {@code R} — max out-degree after robust pruning (must be positive)
   * @param searchListSize Vamana {@code L} — beam width during construction (must be {@code >=
   *     maxDegree})
   * @param alpha diversity parameter (must be {@code >= 1.0}; the Vamana default is {@code 1.2})
   * @param seed random seed for deterministic construction
   * @param threads worker thread count for graph construction (must be {@code >= 1}). Values {@code
   *     > 1} engage {@code ConcurrentVamanaGraphBuilder}; the default {@code 1} preserves byte
   *     identical, deterministic output.
   */
  public record VamanaParams(
      int maxDegree, int searchListSize, float alpha, long seed, int threads) {
    public VamanaParams {
      if (maxDegree <= 0) {
        throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
      }
      if (searchListSize < maxDegree) {
        throw new IllegalArgumentException(
            "searchListSize (" + searchListSize + ") must be >= maxDegree (" + maxDegree + ")");
      }
      if (alpha < 1.0f) {
        throw new IllegalArgumentException("alpha must be >= 1.0: " + alpha);
      }
      if (threads < 1) {
        throw new IllegalArgumentException("threads must be >= 1: " + threads);
      }
    }

    /** Convenience constructor defaulting to single-threaded (deterministic) construction. */
    public VamanaParams(int maxDegree, int searchListSize, float alpha, long seed) {
      this(maxDegree, searchListSize, alpha, seed, 1);
    }
  }

  /**
   * IVF_FLAT build and search parameters.
   *
   * @param k number of clusters (must be positive)
   * @param nprobe number of clusters to probe during search (must be in [1, k])
   * @param maxIter maximum K-Means iterations (must be positive)
   * @param gamma SOAR spill ratio in [0, 1] (0 = no spill)
   * @param soar enable SOAR-style cluster spill during search
   * @param seed RNG seed for K-Means++ initialisation
   */
  public record IvfParams(int k, int nprobe, int maxIter, float gamma, boolean soar, long seed) {
    public IvfParams {
      if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
      if (nprobe <= 0 || nprobe > k)
        throw new IllegalArgumentException("nprobe must be in [1, k]: " + nprobe);
      if (maxIter <= 0) throw new IllegalArgumentException("maxIter must be positive: " + maxIter);
      if (gamma < 0f || gamma > 1f)
        throw new IllegalArgumentException("gamma must be in [0, 1]: " + gamma);
    }

    /** Default: nprobe = max(1, k/4), maxIter = 30, no SOAR, seed = 42. */
    public static IvfParams defaults(int k) {
      return new IvfParams(k, Math.max(1, k / 4), 30, 0f, false, 42L);
    }
  }

  /**
   * IVF-PQ build and search parameters. Extends {@link IvfParams} with product-quantisation
   * settings (subspaces, per-subspace clusters, anisotropic threshold) and a search-time {@code
   * rescoreFactor} that controls the width of the wide candidate heap re-ranked against
   * full-precision vectors.
   *
   * @param k number of IVF clusters (must be positive)
   * @param nprobe number of clusters to probe during search (must be in {@code [1, k]})
   * @param maxIter maximum K-Means iterations (must be positive)
   * @param gamma SOAR spill ratio in {@code [0, 1]} (0 = no spill)
   * @param soar enable SOAR-style cluster spill during IVF search
   * @param seed RNG seed for K-Means++ initialisation
   * @param pqSubspaces number of PQ sub-vectors M (must be positive; must divide dimension)
   * @param pqClusters centroids per subspace Ks (must be in {@code [2, 256]})
   * @param pqAnisotropicThreshold anisotropic threshold for the coordinate-descent encoder; use
   *     {@code -1f} for unweighted (standard) PQ
   * @param rescoreFactor multiplier applied to {@code k} to size the wide candidate heap before
   *     full-precision rescoring (must be {@code >= 1}; {@code 1} disables rescoring)
   */
  public record IvfPqParams(
      int k,
      int nprobe,
      int maxIter,
      float gamma,
      boolean soar,
      long seed,
      int pqSubspaces,
      int pqClusters,
      float pqAnisotropicThreshold,
      int rescoreFactor) {
    public IvfPqParams {
      if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
      if (nprobe <= 0 || nprobe > k)
        throw new IllegalArgumentException("nprobe must be in [1, k]: " + nprobe);
      if (maxIter <= 0) throw new IllegalArgumentException("maxIter must be positive: " + maxIter);
      if (gamma < 0f || gamma > 1f)
        throw new IllegalArgumentException("gamma must be in [0, 1]: " + gamma);
      if (pqSubspaces <= 0)
        throw new IllegalArgumentException("pqSubspaces must be positive: " + pqSubspaces);
      if (pqClusters < 2 || pqClusters > 256)
        throw new IllegalArgumentException("pqClusters must be in [2, 256]: " + pqClusters);
      if (pqAnisotropicThreshold != -1f
          && (pqAnisotropicThreshold < 0f || pqAnisotropicThreshold > 1f))
        throw new IllegalArgumentException(
            "pqAnisotropicThreshold must be -1 (unweighted) or in [0, 1]: "
                + pqAnisotropicThreshold);
      if (rescoreFactor < 1)
        throw new IllegalArgumentException("rescoreFactor must be >= 1: " + rescoreFactor);
    }
  }

  /**
   * NVIDIA cuVS index-build parameters. Sealed so {@link VectorCollectionConfig}'s compact
   * constructor can enforce a one-to-one mapping between {@link IndexType#CUVS_BRUTEFORCE}/{@link
   * IndexType#CUVS_CAGRA} and the variant carried here.
   */
  public sealed interface CuVsParams permits CuVsParams.BruteForce, CuVsParams.Cagra {

    /**
     * GPU brute-force parameters. {@link IndexType#CUVS_BRUTEFORCE} scores every query against
     * every vector on the GPU; no tunables are needed today.
     */
    record BruteForce() implements CuVsParams {
      public static BruteForce defaults() {
        return new BruteForce();
      }
    }

    /**
     * CAGRA graph index parameters.
     *
     * @param graphDegree degree of the final graph (must be positive; FAISS-equivalent to M)
     * @param intermediateGraphDegree degree of the intermediate graph used during build (must be
     *     {@code >= graphDegree})
     * @param itopkSize candidate-list size during search (must be positive; FAISS-equivalent to
     *     efSearch)
     * @param maxIterations max refinement iterations during search (0 = library default)
     */
    record Cagra(int graphDegree, int intermediateGraphDegree, int itopkSize, int maxIterations)
        implements CuVsParams {
      public Cagra {
        if (graphDegree <= 0) {
          throw new IllegalArgumentException("graphDegree must be positive: " + graphDegree);
        }
        if (intermediateGraphDegree < graphDegree) {
          throw new IllegalArgumentException(
              "intermediateGraphDegree ("
                  + intermediateGraphDegree
                  + ") must be >= graphDegree ("
                  + graphDegree
                  + ")");
        }
        if (itopkSize <= 0) {
          throw new IllegalArgumentException("itopkSize must be positive: " + itopkSize);
        }
        if (maxIterations < 0) {
          throw new IllegalArgumentException("maxIterations must be >= 0: " + maxIterations);
        }
      }

      /** Default: graphDegree=32, intermediateGraphDegree=64, itopkSize=64, maxIterations=0. */
      public static Cagra defaults() {
        return new Cagra(32, 64, 64, 0);
      }
    }
  }
}
