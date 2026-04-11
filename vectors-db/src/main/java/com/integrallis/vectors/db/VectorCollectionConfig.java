package com.integrallis.vectors.db;

import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Immutable collection configuration. Captured on {@code build()} and persisted as {@code
 * manifest.bin} in Step 4a.
 *
 * @param dimension fixed vector dimension (must be positive)
 * @param metric similarity function (never null)
 * @param indexType index backend (never null)
 * @param quantizerKind quantizer selector (never null; use {@link QuantizerKind#NONE} if unused)
 * @param autoCommitThreshold if the staging buffer reaches this many documents after an {@code
 *     add}/{@code addAll}, an implicit {@code commit()} runs before the write path returns. Must be
 *     positive. Use {@link Integer#MAX_VALUE} to disable auto-commit entirely (the default).
 * @param storageRoot absolute collection root for persistent mode. When {@code null} the collection
 *     operates as an in-memory-only collection (unchanged Step 3 behaviour for FLAT; Step 4b added
 *     in-memory HNSW support via {@link com.integrallis.vectors.db.index.HnswIndexAdapter}). When
 *     non-null, every {@code commit()} writes a new generation directory under this path via {@link
 *     com.integrallis.vectors.db.storage.GenerationDirectory} and mmaps the result for the next
 *     read snapshot. Step 4b only permits persistent mode with {@link IndexType#FLAT}; persistent
 *     HNSW is deferred to Step 4b Phase 5.
 * @param hnswParams HNSW build-time parameters. Must be {@code non-null iff indexType == HNSW}; any
 *     other combination is rejected by the compact constructor. Added in Step 4b.
 */
public record VectorCollectionConfig(
    int dimension,
    SimilarityFunction metric,
    IndexType indexType,
    QuantizerKind quantizerKind,
    int autoCommitThreshold,
    Path storageRoot,
    HnswParams hnswParams) {

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
  }

  /**
   * 6-arg convenience constructor that defaults {@link #hnswParams()} to {@code null}. Suitable for
   * flat-scan collections. Throws via the compact constructor if the caller asks for {@link
   * IndexType#HNSW} without supplying {@link HnswParams}.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold,
      Path storageRoot) {
    this(dimension, metric, indexType, quantizerKind, autoCommitThreshold, storageRoot, null);
  }

  /**
   * 5-arg convenience constructor for in-memory, flat-scan collections. Equivalent to the 7-arg
   * canonical constructor with {@code null} {@code storageRoot} and {@code null} {@code
   * hnswParams}. Kept for Step 3 test fixtures that pre-date Step 4a/4b.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold) {
    this(dimension, metric, indexType, quantizerKind, autoCommitThreshold, null, null);
  }

  /**
   * HNSW graph-construction parameters captured on {@code VectorCollection.builder().build()}. Only
   * {@code M} is persisted on disk (inside {@code graph.bin}); {@code efConstruction} is a
   * build-time hint that is NOT preserved across a close/reopen, so a reopened collection that
   * triggers another commit will use whichever {@code efConstruction} the caller sets on the new
   * builder invocation.
   *
   * @param m max connections per upper layer (must be positive)
   * @param efConstruction beam width during construction (must be {@code >= m})
   */
  public record HnswParams(int m, int efConstruction) {
    public HnswParams {
      if (m <= 0) {
        throw new IllegalArgumentException("M must be positive: " + m);
      }
      if (efConstruction < m) {
        throw new IllegalArgumentException(
            "efConstruction (" + efConstruction + ") must be >= M (" + m + ")");
      }
    }
  }
}
