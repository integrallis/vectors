/**
 * K-means clustering and centroid-based routing used by the IVF and distributed layers.
 *
 * <p>Key types:
 *
 * <ul>
 *   <li>{@link com.integrallis.vectors.core.cluster.KMeans} — k-means++ training + assignment
 *   <li>{@link com.integrallis.vectors.core.cluster.CentroidIndex} — nearest-centroid routing with
 *       SOAR boundary spill
 * </ul>
 */
package com.integrallis.vectors.core.cluster;
