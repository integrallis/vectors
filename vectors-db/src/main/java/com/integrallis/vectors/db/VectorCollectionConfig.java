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
 */
public record VectorCollectionConfig(
    int dimension, SimilarityFunction metric, IndexType indexType, QuantizerKind quantizerKind) {

  public VectorCollectionConfig {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    Objects.requireNonNull(metric, "metric must not be null");
    Objects.requireNonNull(indexType, "indexType must not be null");
    Objects.requireNonNull(quantizerKind, "quantizerKind must not be null");
  }
}
