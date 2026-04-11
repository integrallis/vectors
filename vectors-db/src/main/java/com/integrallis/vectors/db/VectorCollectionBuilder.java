package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;

/**
 * Fluent builder for {@link VectorCollection}.
 *
 * <p>Required settings: {@link #dimension(int)} and {@link #metric(SimilarityFunction)}. {@link
 * #build()} throws {@link IllegalStateException} if either is unset.
 *
 * <p>Step 4a supports {@link IndexType#FLAT} with {@link QuantizerKind#NONE} in either in-memory
 * mode (no {@link #storagePath(Path)}) or persistent mmap-backed mode (absolute {@code
 * storagePath}).
 */
public final class VectorCollectionBuilder {

  private Integer dimension;
  private SimilarityFunction metric;
  private IndexType indexType = IndexType.FLAT;
  private QuantizerKind quantizerKind = QuantizerKind.NONE;
  private int autoCommitThreshold = Integer.MAX_VALUE;
  private Path storageRoot;

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

  /** Selects the index backend. Step 4a only supports {@link IndexType#FLAT}. */
  public VectorCollectionBuilder indexType(IndexType indexType) {
    if (indexType == null) {
      throw new IllegalArgumentException("indexType must not be null");
    }
    this.indexType = indexType;
    return this;
  }

  /** Selects the quantizer. Step 4a only supports {@link QuantizerKind#NONE}. */
  public VectorCollectionBuilder quantizer(QuantizerKind quantizerKind) {
    if (quantizerKind == null) {
      throw new IllegalArgumentException("quantizerKind must not be null");
    }
    this.quantizerKind = quantizerKind;
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

  /** Builds the collection. Applies Step 4a restrictions on backend and quantizer. */
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
    if (indexType != IndexType.FLAT) {
      throw new UnsupportedOperationException(
          "indexType "
              + indexType
              + " deferred to a later step (Step 4a only supports IndexType.FLAT)");
    }
    if (quantizerKind != QuantizerKind.NONE) {
      throw new UnsupportedOperationException(
          "quantizerKind "
              + quantizerKind
              + " deferred to a later step (Step 4a only supports QuantizerKind.NONE)");
    }
    var config =
        new VectorCollectionConfig(
            dimension, metric, indexType, quantizerKind, autoCommitThreshold, storageRoot);
    return new VectorCollectionImpl(config);
  }
}
