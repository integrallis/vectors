package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.cluster.CentroidIndex;
import com.integrallis.vectors.core.cluster.KMeans;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;

/**
 * Global centroid routing index (Layer 1 of the tiered buoy architecture). Wraps a {@link
 * CentroidIndex} with an optional SOAR spill map and cluster metadata.
 *
 * <p>Always resident in JVM heap. Immutable once built; rebuilt when the cluster topology changes.
 * Memory footprint at K=1024, D=128: ~524 KB (fits in L2 cache).
 */
public final class BuoyIndex {

  private final float[][] buoyVectors;
  private final int[] spillTargets; // spillTargets[i] = nearest neighbour cluster of cluster i
  private final int[] clusterSizes;
  private final float[] clusterRadii;
  private final SimilarityFunction metric;
  private final CentroidIndex centroidIndex;

  private BuoyIndex(
      float[][] buoyVectors,
      int[] spillTargets,
      int[] clusterSizes,
      float[] clusterRadii,
      SimilarityFunction metric) {
    this.buoyVectors = buoyVectors;
    this.spillTargets = spillTargets;
    this.clusterSizes = clusterSizes;
    this.clusterRadii = clusterRadii;
    this.metric = metric;
    this.centroidIndex = new CentroidIndex(buoyVectors, metric);
  }

  /**
   * Trains {@code k} buoys on {@code dataset}. When {@code buildSoar=true}, constructs the spill
   * map by finding each training vector's second-nearest centroid (SOAR construction).
   *
   * @param dataset training vectors [n][dim]; n &ge; 256*k recommended
   * @param k number of buoys (clusters)
   * @param metric similarity function
   * @param buildSoar whether to compute SOAR spill targets
   * @param seed RNG seed
   */
  public static BuoyIndex train(
      float[][] dataset, int k, SimilarityFunction metric, boolean buildSoar, long seed) {
    float[][] centroids = KMeans.train(dataset, k, 50, seed);
    int[] assignments = KMeans.assign(dataset, centroids);

    int[] sizes = new int[k];
    for (int a : assignments) sizes[a]++;

    float[] radii = computeRadii(dataset, centroids, assignments);
    int[] spill = buildSoar ? computeSpillTargets(dataset, centroids, metric) : allMinus1(k);

    return new BuoyIndex(centroids, spill, sizes, radii, metric);
  }

  /**
   * Routes {@code query} to the {@code nprobe} nearest clusters. When {@code gamma > 0}, SOAR
   * boundary expansion adds spill targets for clusters within {@code (1+gamma)*nearestDistance}.
   */
  public int[] route(float[] query, int nprobe, float gamma) {
    return centroidIndex.routeWithSpill(query, nprobe, gamma, spillTargets);
  }

  public int k() {
    return buoyVectors.length;
  }

  public SimilarityFunction metric() {
    return metric;
  }

  /** Returns a defensive copy of the buoy (centroid) vectors. */
  public float[][] buoyVectors() {
    float[][] copy = new float[buoyVectors.length][];
    for (int i = 0; i < buoyVectors.length; i++) copy[i] = buoyVectors[i].clone();
    return copy;
  }

  /** Returns a defensive copy of the SOAR spill targets. */
  public int[] spillTargets() {
    return spillTargets.clone();
  }

  /** Returns a defensive copy of cluster sizes. */
  public int[] clusterSizes() {
    return clusterSizes.clone();
  }

  /** Returns a defensive copy of cluster radii. */
  public float[] clusterRadii() {
    return clusterRadii.clone();
  }

  /** Encodes this index to bytes for persistence or gossip propagation. */
  public byte[] encode() {
    try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bos)) {
      int k = buoyVectors.length;
      int dim = buoyVectors[0].length;
      out.writeInt(k);
      out.writeInt(dim);
      out.writeUTF(metric.name());
      for (float[] row : buoyVectors) for (float v : row) out.writeFloat(v);
      for (int s : spillTargets) out.writeInt(s);
      for (int s : clusterSizes) out.writeInt(s);
      for (float r : clusterRadii) out.writeFloat(r);
      out.flush();
      return bos.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /** Restores a {@link BuoyIndex} from bytes produced by {@link #encode()}. */
  public static BuoyIndex decode(byte[] bytes) {
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      int k = in.readInt();
      int dim = in.readInt();
      SimilarityFunction metric = SimilarityFunction.valueOf(in.readUTF());
      float[][] buoys = new float[k][dim];
      for (float[] row : buoys) for (int d = 0; d < dim; d++) row[d] = in.readFloat();
      int[] spill = new int[k];
      for (int i = 0; i < k; i++) spill[i] = in.readInt();
      int[] sizes = new int[k];
      for (int i = 0; i < k; i++) sizes[i] = in.readInt();
      float[] radii = new float[k];
      for (int i = 0; i < k; i++) radii[i] = in.readFloat();
      return new BuoyIndex(buoys, spill, sizes, radii, metric);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  // --- internals ---

  /**
   * Computes the maximum Euclidean distance from each centroid to any vector assigned to it. Always
   * uses Euclidean distance regardless of the similarity function — radii measure geometric cluster
   * extent for metadata and split decisions, not for routing.
   */
  private static float[] computeRadii(float[][] dataset, float[][] centroids, int[] assignments) {
    int k = centroids.length;
    float[] radii = new float[k];
    for (int i = 0; i < dataset.length; i++) {
      float d = euclideanDist(dataset[i], centroids[assignments[i]]);
      if (d > radii[assignments[i]]) radii[assignments[i]] = d;
    }
    return radii;
  }

  /**
   * Computes SOAR spill targets using frequency-based selection: for each cluster, the spill target
   * is the second-nearest centroid most frequently observed across all training vectors assigned to
   * that cluster.
   */
  private static int[] computeSpillTargets(
      float[][] dataset, float[][] centroids, SimilarityFunction metric) {
    int k = centroids.length;
    CentroidIndex ci = new CentroidIndex(centroids, metric);
    // freq[i][j] = number of vectors in cluster i whose second-nearest centroid is j
    int[][] freq = new int[k][k];
    for (float[] v : dataset) {
      int[] top2 = ci.route(v, Math.min(2, k));
      if (top2.length >= 2) {
        freq[top2[0]][top2[1]]++;
      }
    }
    // For each cluster, pick the second-nearest centroid with the highest frequency
    int[] spill = allMinus1(k);
    for (int i = 0; i < k; i++) {
      int bestTarget = -1;
      int bestCount = 0;
      for (int j = 0; j < k; j++) {
        if (j == i) continue;
        if (freq[i][j] > bestCount) {
          bestCount = freq[i][j];
          bestTarget = j;
        }
      }
      spill[i] = bestTarget;
    }
    return spill;
  }

  private static float euclideanDist(float[] a, float[] b) {
    float sum = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      sum += d * d;
    }
    return (float) Math.sqrt(sum);
  }

  private static int[] allMinus1(int k) {
    int[] a = new int[k];
    Arrays.fill(a, -1);
    return a;
  }
}
