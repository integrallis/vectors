package com.integrallis.vectors.ivf;

/**
 * Immutable posting list for a single IVF cluster: its centroid plus the ordered list of vector
 * ordinals whose owning vectors were assigned to this cluster at build time.
 *
 * <p>Ordinals are in the global ordinal space of the source vector set (i.e., the row index in the
 * {@code float[][] vectors} array passed to {@link IvfIndex#build}), not cluster-local positions.
 *
 * <p><b>HARMONY partial-distance pruning</b>: when {@link #keyDimensions} is non-null (set at build
 * time via {@link IvfBuildParams#harmonyKeyDims}), the {@link IvfIndex} scan loop first computes a
 * partial squared-Euclidean distance over only those dimensions. Because partial ≤ full for L2, the
 * partial distance is a valid lower bound; candidates whose lower bound already exceeds the current
 * k-th neighbour distance are skipped without computing the full distance.
 *
 * @param clusterId unique cluster identifier (0-based index into the BuoyIndex)
 * @param centroid centroid vector; same dimension as the indexed vectors
 * @param ordinals global ordinals of the assigned vectors (may be empty)
 * @param size {@code ordinals.length}
 * @param keyDimensions top-variance dimension indices for partial-distance pruning; {@code null}
 *     when HARMONY is disabled
 */
public record ClusterPartition(
    int clusterId, float[] centroid, int[] ordinals, int size, int[] keyDimensions) {

  /** Validates that size matches ordinals.length. */
  public ClusterPartition {
    if (ordinals == null) throw new IllegalArgumentException("ordinals must not be null");
    if (size != ordinals.length) {
      throw new IllegalArgumentException(
          "size (" + size + ") must equal ordinals.length (" + ordinals.length + ")");
    }
    // keyDimensions is nullable — null means HARMONY pruning is disabled for this partition
  }

  /**
   * Convenience factory that creates a partition without HARMONY key dimensions (pruning disabled).
   * Use when {@link IvfBuildParams#harmonyKeyDims} is 0.
   */
  public static ClusterPartition of(int clusterId, float[] centroid, int[] ordinals) {
    return new ClusterPartition(clusterId, centroid, ordinals, ordinals.length, null);
  }

  /** Returns {@code true} when HARMONY partial-distance pruning is enabled for this partition. */
  public boolean hasKeyDimensions() {
    return keyDimensions != null && keyDimensions.length > 0;
  }

  /** Returns {@code true} when no vectors are assigned to this cluster. */
  public boolean isEmpty() {
    return size == 0;
  }
}
