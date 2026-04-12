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
 *     in-memory HNSW support via {@link com.integrallis.vectors.db.index.HnswIndexAdapter} and Step
 *     4c added in-memory VAMANA support via {@link
 *     com.integrallis.vectors.db.index.VamanaIndexAdapter}). When non-null, every {@code commit()}
 *     writes a new generation directory under this path via {@link
 *     com.integrallis.vectors.db.storage.GenerationDirectory} and mmaps the result for the next
 *     read snapshot.
 * @param hnswParams HNSW build-time parameters. Must be {@code non-null iff indexType == HNSW}; any
 *     other combination is rejected by the compact constructor. Added in Step 4b.
 * @param vamanaParams Vamana build-time parameters. Must be {@code non-null iff indexType ==
 *     VAMANA}; any other combination is rejected by the compact constructor. Added in Step 4c.
 * @param quantizerParams build-time quantizer parameters. Must be {@code null} when {@code
 *     quantizerKind == NONE}. When non-NONE, may be {@code null} to use defaults. When non-null,
 *     the record type must match the quantizer kind. Added in Step 4d.
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
    QuantizerParams quantizerParams) {

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
  }

  /**
   * 8-arg convenience constructor that defaults {@link #quantizerParams()} to {@code null}.
   * Preserves the Step 4c canonical shape for call sites that predate Step 4d.
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
        null);
  }

  /**
   * 7-arg convenience constructor that defaults {@link #vamanaParams()} and {@link
   * #quantizerParams()} to {@code null}. Preserves the Step 4b canonical shape for call sites that
   * predate Step 4c. Throws via the compact constructor if the caller asks for {@link
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
        null);
  }

  /**
   * 5-arg convenience constructor for in-memory, flat-scan collections. Equivalent to the 9-arg
   * canonical constructor with {@code null} {@code storageRoot}, {@code null} {@code hnswParams},
   * {@code null} {@code vamanaParams}, and {@code null} {@code quantizerParams}. Kept for Step 3
   * test fixtures that pre-date Step 4a/4b.
   */
  public VectorCollectionConfig(
      int dimension,
      SimilarityFunction metric,
      IndexType indexType,
      QuantizerKind quantizerKind,
      int autoCommitThreshold) {
    this(dimension, metric, indexType, quantizerKind, autoCommitThreshold, null, null, null, null);
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
   */
  public record VamanaParams(int maxDegree, int searchListSize, float alpha, long seed) {
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
    }
  }
}
