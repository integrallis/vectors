package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.io.Closeable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * IVF-flat index: route query to {@code nprobe} clusters via {@link BuoyIndex}, then brute-force
 * scan each cluster's vectors, merge top-k.
 *
 * <p>This is the Phase 1 search target and the simplest correct IVF implementation. Builds entirely
 * in-heap from raw float arrays; no dependency on {@code vectors-db}.
 *
 * <p>Build complexity: O(n * k * maxIter). Search complexity: O(nprobe * clusterSize * D).
 */
public final class IvfIndex implements Closeable {

  private final BuoyIndex buoyIndex;
  private final ClusterPartition[] partitions;
  private final float[][] vectors; // all indexed vectors [n][dim]
  private final String[] ids; // optional document ids; may be null entries
  private final SimilarityFunction metric;

  private IvfIndex(
      BuoyIndex buoyIndex,
      ClusterPartition[] partitions,
      float[][] vectors,
      String[] ids,
      SimilarityFunction metric) {
    this.buoyIndex = buoyIndex;
    this.partitions = partitions;
    this.vectors = vectors;
    this.ids = ids;
    this.metric = metric;
  }

  /**
   * Builds an IVF-flat index from {@code vectors} using {@code params}.
   *
   * @param vectors input vectors [n][dim]; not modified
   * @param ids optional document identifiers (null = no ids); if non-null must have length n
   * @param metric similarity function
   * @param params build parameters
   */
  public static IvfIndex build(
      float[][] vectors, String[] ids, SimilarityFunction metric, IvfBuildParams params) {
    if (vectors.length == 0) throw new IllegalArgumentException("empty vector set");
    int n = vectors.length;
    int k = Math.min(params.k(), n);

    BuoyIndex buoy = BuoyIndex.train(vectors, k, metric, params.buildSoar(), params.seed());
    int[] assignments = assignAll(vectors, buoy);

    // Build posting lists
    List<List<Integer>> lists = new ArrayList<>(k);
    for (int c = 0; c < k; c++) lists.add(new ArrayList<>());
    for (int i = 0; i < n; i++) lists.get(assignments[i]).add(i);

    ClusterPartition[] partitions = new ClusterPartition[k];
    float[][] buoys = buoy.buoyVectors();
    for (int c = 0; c < k; c++) {
      int[] ordinals = lists.get(c).stream().mapToInt(Integer::intValue).toArray();
      partitions[c] = new ClusterPartition(c, buoys[c], ordinals, ordinals.length);
    }

    return new IvfIndex(buoy, partitions, vectors, ids, metric);
  }

  /**
   * Searches the index for the top-k nearest neighbours of {@code request.query()}.
   *
   * <p>Route → brute-force per cluster → merge top-k across clusters.
   */
  public IvfSearchResult search(IvfSearchRequest request) {
    float[] query = request.query();
    int k = request.k();
    int nprobe = Math.min(request.nprobe(), partitions.length);
    int[] clusterIds = buoyIndex.route(query, nprobe, request.gamma());

    // Min-heap keyed on score descending: we want the top-k highest scores.
    // Use a min-heap of size k: evict the smallest when heap is full.
    PriorityQueue<IvfHit> heap =
        new PriorityQueue<>(k + 1, (a, b) -> Float.compare(a.score(), b.score()));

    for (int cid : clusterIds) {
      ClusterPartition partition = partitions[cid];
      for (int ordinal : partition.ordinals()) {
        float score = score(query, vectors[ordinal]);
        if (score < request.minScore()) continue;
        if (heap.size() < k) {
          heap.offer(new IvfHit(ordinal, ids != null ? ids[ordinal] : null, score));
        } else if (score > heap.peek().score()) {
          heap.poll();
          heap.offer(new IvfHit(ordinal, ids != null ? ids[ordinal] : null, score));
        }
      }
    }

    // Drain heap into descending-score list
    IvfHit[] arr = heap.toArray(new IvfHit[0]);
    Arrays.sort(arr); // IvfHit.compareTo is descending by score
    return new IvfSearchResult(List.of(arr), clusterIds.length);
  }

  public BuoyIndex buoyIndex() {
    return buoyIndex;
  }

  public int clusterCount() {
    return partitions.length;
  }

  public ClusterPartition partition(int clusterId) {
    return partitions[clusterId];
  }

  @Override
  public void close() {
    // no off-heap resources in Phase 1
  }

  // --- internals ---

  private float score(float[] query, float[] vector) {
    return switch (metric) {
      case EUCLIDEAN -> -VectorUtil.squareDistance(query, vector); // negate: higher = nearer
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> VectorUtil.dotProduct(query, vector);
      case COSINE -> VectorUtil.cosine(query, vector);
    };
  }

  private static int[] assignAll(float[][] vectors, BuoyIndex buoy) {
    int n = vectors.length;
    int[] out = new int[n];
    for (int i = 0; i < n; i++) {
      int[] nearest = buoy.route(vectors[i], 1, 0f);
      out[i] = nearest[0];
    }
    return out;
  }
}
