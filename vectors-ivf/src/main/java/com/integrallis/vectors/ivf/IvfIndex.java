package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

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
   * <p>Route → bulk SIMD scan per cluster (fused GEMV, 4-row unrolled) → primitive top-k heap. When
   * HARMONY key-dimension pruning is active and the heap is saturated, falls back to a scalar path
   * within that partition so the partial-distance lower bound stays on the hot path.
   *
   * <p>For EUCLIDEAN, clusters are visited in ascending centroid distance and a triangle-inequality
   * lower bound ({@code sqrt(centroidSqDist) - radius}) is compared against the current worst top-k
   * distance; the loop terminates early once even the nearest unscanned cluster cannot contribute a
   * competing result.
   */
  public IvfSearchResult search(IvfSearchRequest request) {
    float[] query = request.query();
    int k = request.k();
    int nprobe = Math.min(request.nprobe(), partitions.length);
    var route = buoyIndex.routeWithDistances(query, nprobe, request.gamma());
    int[] clusterIds = route.clusterIds();
    float[] centroidDists = route.distances();
    float[] radii = buoyIndex.clusterRadii();

    TopKHeap heap = new TopKHeap(k);
    boolean euclidean = metric == SimilarityFunction.EUCLIDEAN;
    float minScore = request.minScore();

    // Reusable scratch buffers for batch SIMD scan.
    float[][] batchRows = new float[BATCH_ROWS][];
    float[] batchOut = new float[BATCH_ROWS];

    int clustersScanned = 0;
    for (int i = 0; i < clusterIds.length; i++) {
      int cid = clusterIds[i];
      // Triangle-inequality lower-bound pruning. Applicable only for EUCLIDEAN where
      // centroidDists[] are squared L2 distances and radii are L2 distances. score = -sqDist,
      // so the heap's worst entry has score = -worstSqDist; lowerBound on sqDist for any v in
      // this cluster is (max(0, sqrt(centroidSqDist) - radius))^2.
      if (euclidean && heap.isFull()) {
        float centroidL2 = (float) Math.sqrt(centroidDists[i]);
        float lb = centroidL2 - radii[cid];
        if (lb > 0f) {
          float lbSq = lb * lb;
          if (lbSq >= -heap.worst()) break; // route() sorted ascending, so no later cluster fits
        }
      }

      ClusterPartition partition = partitions[cid];
      int[] ords = partition.ordinals();
      if (ords.length == 0) {
        clustersScanned++;
        continue;
      }

      boolean prune = euclidean && partition.hasKeyDimensions();
      if (prune && heap.isFull()) {
        scanPartitionWithHarmony(query, ords, partition.keyDimensions(), minScore, heap);
      } else {
        scanPartitionBulk(query, ords, minScore, heap, batchRows, batchOut);
      }
      clustersScanned++;
    }

    TopKHeap.DrainResult drained = heap.drainDescending();
    List<IvfHit> hits = new ArrayList<>(drained.ordinals().length);
    for (int i = 0; i < drained.ordinals().length; i++) {
      int ord = drained.ordinals()[i];
      hits.add(new IvfHit(ord, ids != null ? ids[ord] : null, drained.scores()[i]));
    }
    return new IvfSearchResult(hits, clustersScanned);
  }

  /** Batch size for fused GEMV scan. Matched to {@link VectorUtil#batchSquaredL2} unroll factor. */
  private static final int BATCH_ROWS = 64;

  /**
   * Fused 4-row SIMD scan over a partition. No per-row allocation, no pruning. COSINE falls back to
   * scalar since there is no fused GEMV cosine kernel yet.
   */
  private void scanPartitionBulk(
      float[] query, int[] ords, float minScore, TopKHeap heap, float[][] rows, float[] out) {
    int n = ords.length;
    if (metric == SimilarityFunction.COSINE) {
      for (int ord : ords) {
        float score = score(query, vectors[ord]);
        if (score < minScore) continue;
        heap.offer(ord, score);
      }
      return;
    }
    boolean euclidean = metric == SimilarityFunction.EUCLIDEAN;
    for (int start = 0; start < n; start += BATCH_ROWS) {
      int count = Math.min(BATCH_ROWS, n - start);
      for (int j = 0; j < count; j++) rows[j] = vectors[ords[start + j]];
      if (euclidean) {
        VectorUtil.batchSquaredL2(query, rows, out, count);
      } else {
        VectorUtil.batchDotProduct(query, rows, out, count);
      }
      for (int j = 0; j < count; j++) {
        float score = euclidean ? -out[j] : out[j];
        if (score < minScore) continue;
        heap.offer(ords[start + j], score);
      }
    }
  }

  /** Scalar path preserving HARMONY partial-distance pruning. */
  private void scanPartitionWithHarmony(
      float[] query, int[] ords, int[] keyDims, float minScore, TopKHeap heap) {
    for (int ordinal : ords) {
      // Pruning is valid only for EUCLIDEAN (partial sum ≤ full sum).
      float partialSq = DimensionAnalysis.partialSquaredDistance(query, vectors[ordinal], keyDims);
      if (partialSq >= -heap.worst()) continue;
      float score = score(query, vectors[ordinal]);
      if (score < minScore) continue;
      heap.offer(ordinal, score);
    }
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
   * <p>Format: {@code [buoyLen:4][buoyBytes][k:4]
   * k×([ordCount:4][ordinals:4*n][keyDimCount:4][keyDims:4*m])} where {@code keyDimCount=0} means
   * HARMONY pruning is disabled for that partition.
   */
  public byte[] encode() {
    byte[] buoyBytes = buoyIndex.encode();
    int k = partitions.length;

    int totalOrds = 0;
    int totalKeyDims = 0;
    for (ClusterPartition p : partitions) {
      totalOrds += p.size();
      if (p.keyDimensions() != null) totalKeyDims += p.keyDimensions().length;
    }
    // Each partition: ordCount(4) + ords(4*n) + keyDimCount(4) + keyDims(4*m)
    int capacity = 4 + buoyBytes.length + 4 + k * 8 + totalOrds * 4 + totalKeyDims * 4;

    ByteBuffer buf = ByteBuffer.allocate(capacity);
    buf.putInt(buoyBytes.length);
    buf.put(buoyBytes);
    buf.putInt(k);
    for (ClusterPartition p : partitions) {
      buf.putInt(p.size());
      for (int ord : p.ordinals()) buf.putInt(ord);
      int[] kd = p.keyDimensions();
      if (kd == null) {
        buf.putInt(0);
      } else {
        buf.putInt(kd.length);
        for (int d : kd) buf.putInt(d);
      }
    }
    return buf.array();
  }

  /**
   * Deserialises a previously {@link #encode encoded} index, wiring it to {@code vectors} and
   * {@code metric} without re-running KMeans. HARMONY {@code keyDimensions} are restored so
   * partial-distance pruning is active immediately after decode.
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
      int kdLen = buf.getInt();
      int[] keyDims = kdLen == 0 ? null : new int[kdLen];
      if (keyDims != null) for (int i = 0; i < kdLen; i++) keyDims[i] = buf.getInt();
      partitions[c] = new ClusterPartition(c, centroids[c], ordinals, size, keyDims);
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
