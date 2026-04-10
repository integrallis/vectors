package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Fluent builder for {@link VectorCollection}.
 *
 * <p>Required settings: {@link #dimension(int)} and {@link #metric(SimilarityFunction)}. {@link
 * #build()} throws {@link IllegalStateException} if either is unset.
 *
 * <p>Step 2 only supports {@link IndexType#FLAT} with {@link QuantizerKind#NONE}; other backends
 * throw {@link UnsupportedOperationException} at {@link #build()} time.
 */
public final class VectorCollectionBuilder {

  private Integer dimension;
  private SimilarityFunction metric;
  private IndexType indexType = IndexType.FLAT;
  private QuantizerKind quantizerKind = QuantizerKind.NONE;
  private int autoCommitThreshold = Integer.MAX_VALUE;

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

  /** Selects the index backend. Step 2 only supports {@link IndexType#FLAT}. */
  public VectorCollectionBuilder indexType(IndexType indexType) {
    if (indexType == null) {
      throw new IllegalArgumentException("indexType must not be null");
    }
    this.indexType = indexType;
    return this;
  }

  /** Selects the quantizer. Step 2 only supports {@link QuantizerKind#NONE}. */
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

  /** Builds the collection. Applies Step 2 restrictions on backend and quantizer. */
  public VectorCollection build() {
    if (dimension == null) {
      throw new IllegalStateException("dimension is required, call builder.dimension(d)");
    }
    if (metric == null) {
      throw new IllegalStateException("metric is required, call builder.metric(m)");
    }
    if (indexType != IndexType.FLAT) {
      throw new UnsupportedOperationException(
          "indexType "
              + indexType
              + " deferred to a later step (Step 2 only supports IndexType.FLAT)");
    }
    if (quantizerKind != QuantizerKind.NONE) {
      throw new UnsupportedOperationException(
          "quantizerKind "
              + quantizerKind
              + " deferred to a later step (Step 2 only supports QuantizerKind.NONE)");
    }
    var config =
        new VectorCollectionConfig(
            dimension, metric, indexType, quantizerKind, autoCommitThreshold);
    return new VectorCollectionImpl(config);
  }
}
