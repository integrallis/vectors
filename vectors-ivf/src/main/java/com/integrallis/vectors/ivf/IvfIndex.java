package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.io.Closeable;
import java.nio.ByteBuffer;
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

    float[][] buoys = buoy.buoyVectors();
    ClusterPartition[] partitions = new ClusterPartition[k];
    int harmonyDims = params.harmonyKeyDims();
    for (int c = 0; c < k; c++) {
      int[] ordinals = lists.get(c).stream().mapToInt(Integer::intValue).toArray();
      int[] keyDims = null;
      if (harmonyDims > 0 && ordinals.length > 0) {
        float[][] subVectors = new float[ordinals.length][];
        for (int i = 0; i < ordinals.length; i++) subVectors[i] = vectors[ordinals[i]];
        keyDims = DimensionAnalysis.topVarianceDimensions(subVectors, harmonyDims);
      }
      partitions[c] = new ClusterPartition(c, buoys[c], ordinals, ordinals.length, keyDims);
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

    boolean harmonyEuclidean = metric == SimilarityFunction.EUCLIDEAN;
    for (int cid : clusterIds) {
      ClusterPartition partition = partitions[cid];
      boolean prune = harmonyEuclidean && partition.hasKeyDimensions();
      int[] keyDims = prune ? partition.keyDimensions() : null;
      for (int ordinal : partition.ordinals()) {
        // HARMONY partial-distance lower bound: skip if partial L2 already exceeds worst result.
        // Valid only for EUCLIDEAN (partial sum ≤ full sum for any dimension subset).
        if (prune && heap.size() >= k) {
          float partialSq =
              DimensionAnalysis.partialSquaredDistance(query, vectors[ordinal], keyDims);
          // score = -fullSqDist; worstScore = heap.peek().score() (negative).
          // Prune when fullSqDist >= -worstScore, i.e. partialSqDist >= -worstScore.
          if (partialSq >= -heap.peek().score()) continue;
        }
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

  /** Returns the total number of indexed vectors. */
  public int size() {
    return vectors.length;
  }

  /** Returns the number of clusters (K). */
  public int k() {
    return partitions.length;
  }

  public BuoyIndex buoyIndex() {
    return buoyIndex;
  }

  public ClusterPartition partition(int clusterId) {
    return partitions[clusterId];
  }

  /**
   * Constructs an {@link IvfIndex} from pre-trained components — used by codec deserialisers to
   * avoid re-running KMeans on load.
   */
  public static IvfIndex fromPrebuilt(
      BuoyIndex buoyIndex,
      ClusterPartition[] partitions,
      float[][] vectors,
      SimilarityFunction metric) {
    return new IvfIndex(buoyIndex, partitions, vectors, null, metric);
  }

  /**
   * Serialises this index to a compact byte array.
   *
   * <p>Format: {@code [buoyLen:4][buoyBytes][k:4] k×([ordCount:4][ordinals:4*n])}
   */
  public byte[] encode() {
    byte[] buoyBytes = buoyIndex.encode();
    int k = partitions.length;

    int totalOrds = 0;
    for (ClusterPartition p : partitions) totalOrds += p.size();
    int capacity = 4 + buoyBytes.length + 4 + k * 4 + totalOrds * 4;

    ByteBuffer buf = ByteBuffer.allocate(capacity);
    buf.putInt(buoyBytes.length);
    buf.put(buoyBytes);
    buf.putInt(k);
    for (ClusterPartition p : partitions) {
      buf.putInt(p.size());
      for (int ord : p.ordinals()) buf.putInt(ord);
    }
    return buf.array();
  }

  /**
   * Deserialises a previously {@link #encode encoded} index, wiring it to {@code vectors} and
   * {@code metric} without re-running KMeans.
   */
  public static IvfIndex decode(byte[] bytes, float[][] vectors, SimilarityFunction metric) {
    ByteBuffer buf = ByteBuffer.wrap(bytes);

    int buoyLen = buf.getInt();
    byte[] buoyBytes = new byte[buoyLen];
    buf.get(buoyBytes);
    BuoyIndex buoyIndex = BuoyIndex.decode(buoyBytes);

    int k = buf.getInt();
    float[][] centroids = buoyIndex.buoyVectors();
    ClusterPartition[] partitions = new ClusterPartition[k];
    for (int c = 0; c < k; c++) {
      int size = buf.getInt();
      int[] ordinals = new int[size];
      for (int i = 0; i < size; i++) ordinals[i] = buf.getInt();
      partitions[c] = ClusterPartition.of(c, centroids[c], ordinals);
    }
    return new IvfIndex(buoyIndex, partitions, vectors, null, metric);
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
