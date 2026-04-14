package com.integrallis.vectors.ivf;

/**
 * Immutable posting list for a single IVF cluster: its centroid plus the ordered list of vector
 * ordinals whose owning vectors were assigned to this cluster at build time.
 *
 * <p>Ordinals are in the global ordinal space of the source vector set (i.e., the row index in the
 * {@code float[][] vectors} array passed to {@link IvfIndex#build}), not cluster-local positions.
 *
 * @param clusterId unique cluster identifier (0-based index into the BuoyIndex)
 * @param centroid centroid vector; same dimension as the indexed vectors
 * @param ordinals global ordinals of the assigned vectors (may be empty)
 * @param size {@code ordinals.length}
 */
public record ClusterPartition(int clusterId, float[] centroid, int[] ordinals, int size) {

  /** Returns {@code true} when no vectors are assigned to this cluster. */
  public boolean isEmpty() {
    return size == 0;
  }
}
