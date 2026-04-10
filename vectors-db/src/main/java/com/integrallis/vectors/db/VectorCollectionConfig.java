package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Objects;

/**
 * Immutable collection configuration. Captured on {@code build()} and persisted as {@code
 * manifest.json} in later steps.
 *
 * @param dimension fixed vector dimension (must be positive)
 * @param metric similarity function (never null)
 * @param indexType index backend (never null)
 * @param quantizerKind quantizer selector (never null; use {@link QuantizerKind#NONE} if unused)
 * @param autoCommitThreshold if the staging buffer reaches this many documents after an {@code
 *     add}/{@code addAll}, an implicit {@code commit()} runs before the write path returns. Must be
 *     positive. Use {@link Integer#MAX_VALUE} to disable auto-commit entirely (the default).
 */
public record VectorCollectionConfig(
    int dimension,
    SimilarityFunction metric,
    IndexType indexType,
    QuantizerKind quantizerKind,
    int autoCommitThreshold) {

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
  }
}
