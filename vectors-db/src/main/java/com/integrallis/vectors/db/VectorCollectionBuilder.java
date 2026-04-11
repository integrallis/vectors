package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;

/**
 * Fluent builder for {@link VectorCollection}.
 *
 * <p>Required settings: {@link #dimension(int)} and {@link #metric(SimilarityFunction)}. {@link
 * #build()} throws {@link IllegalStateException} if either is unset.
 *
 * <p>Step 4c supports {@link IndexType#FLAT}, {@link IndexType#HNSW}, and {@link IndexType#VAMANA}
 * with {@link QuantizerKind#NONE} in either in-memory mode (no {@link #storagePath(Path)}) or
 * persistent mmap-backed mode (absolute {@code storagePath}). Non-{@link QuantizerKind#NONE}
 * quantizers are deferred to Step 4d.
 */
public final class VectorCollectionBuilder {

  /** HNSW {@code M} parameter default. Matches the HnswIndex.Builder default. */
  public static final int DEFAULT_HNSW_M = 16;

  /** HNSW {@code efConstruction} parameter default. Matches the HnswIndex.Builder default. */
  public static final int DEFAULT_HNSW_EF_CONSTRUCTION = 200;

  /** Vamana {@code R} (maxDegree) parameter default. Matches the VamanaIndex.Builder default. */
  public static final int DEFAULT_VAMANA_R = 64;

  /**
   * Vamana {@code L} (searchListSize) parameter default. Matches the VamanaIndex.Builder default.
   */
  public static final int DEFAULT_VAMANA_L = 128;

  /** Vamana {@code alpha} parameter default. Matches the VamanaIndex.Builder default. */
  public static final float DEFAULT_VAMANA_ALPHA = 1.2f;

  private Integer dimension;
  private SimilarityFunction metric;
  private IndexType indexType = IndexType.FLAT;
  private QuantizerKind quantizerKind = QuantizerKind.NONE;
  private int autoCommitThreshold = Integer.MAX_VALUE;
  private Path storageRoot;
  private int hnswM = DEFAULT_HNSW_M;
  private int hnswEfConstruction = DEFAULT_HNSW_EF_CONSTRUCTION;
  private int vamanaMaxDegree = DEFAULT_VAMANA_R;
  private int vamanaSearchListSize = DEFAULT_VAMANA_L;
  private float vamanaAlpha = DEFAULT_VAMANA_ALPHA;
  private Long vamanaSeed; // lazily filled with System.nanoTime() at build() time if unset

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

  /**
   * Selects the index backend. Step 4c supports {@link IndexType#FLAT}, {@link IndexType#HNSW}, and
   * {@link IndexType#VAMANA}.
   */
  public VectorCollectionBuilder indexType(IndexType indexType) {
    if (indexType == null) {
      throw new IllegalArgumentException("indexType must not be null");
    }
    this.indexType = indexType;
    return this;
  }

  /** Selects the quantizer. Step 4c only supports {@link QuantizerKind#NONE}. */
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

  /** Builds the collection. Applies Step 4c restrictions on backend and quantizer. */
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
    if (quantizerKind != QuantizerKind.NONE) {
      throw new UnsupportedOperationException(
          "quantizerKind "
              + quantizerKind
              + " deferred to a later step (Step 4c only supports QuantizerKind.NONE)");
    }
    VectorCollectionConfig.HnswParams hnswParams =
        (indexType == IndexType.HNSW)
            ? new VectorCollectionConfig.HnswParams(hnswM, hnswEfConstruction)
            : null;
    VectorCollectionConfig.VamanaParams vamanaParams =
        (indexType == IndexType.VAMANA)
            ? new VectorCollectionConfig.VamanaParams(
                vamanaMaxDegree,
                vamanaSearchListSize,
                vamanaAlpha,
                vamanaSeed != null ? vamanaSeed : System.nanoTime())
            : null;
    var config =
        new VectorCollectionConfig(
            dimension,
            metric,
            indexType,
            quantizerKind,
            autoCommitThreshold,
            storageRoot,
            hnswParams,
            vamanaParams);
    return new VectorCollectionImpl(config);
  }
}
