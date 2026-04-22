package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ProductQuantizer;
import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * IVF index with optional product quantization: route query to {@code nprobe} clusters via {@link
 * BuoyIndex}, then scan each cluster's vectors, merge top-k.
 *
 * <p>Two modes, selected at build time via {@link IvfBuildParams#pqEnabled()}:
 *
 * <ul>
 *   <li><b>IVF-flat</b>: brute-force full-precision scan per cluster. Simplest correct IVF
 *       implementation. Build complexity: {@code O(n * k * maxIter)}; search complexity: {@code
 *       O(nprobe * clusterSize * D)}.
 *   <li><b>IVF-PQ</b>: a {@link ProductQuantizer} is trained on per-cluster residuals ({@code v -
 *       centroid(assigned cluster)}). Each vector is encoded into an {@code M}-byte code. At search
 *       time, for each probed cluster the residual query {@code q - centroid(c)} drives an {@link
 *       ProductQuantizer#buildADCTable ADC lookup table}, and ordinals in the partition are scored
 *       via {@code M} table lookups per ordinal (asymmetric distance computation). An optional
 *       full-precision rescore pass re-ranks the top {@code k * rescoreFactor} ADC candidates.
 * </ul>
 *
 * <p>Full-precision vectors are retained in-heap in both modes to support rescoring and to keep the
 * codec compatible across modes. Pure-PQ storage (code-only) is a future extension.
 *
 * <p>HARMONY partial-distance pruning is silently ignored when PQ is enabled.
 */
public final class IvfIndex implements Closeable {

  private final BuoyIndex buoyIndex;
  private final ClusterPartition[] partitions;
  private final float[][] vectors; // all indexed vectors [n][dim]
  private final String[] ids; // optional document ids; may be null entries
  private final SimilarityFunction metric;
  // PQ state; both null when the index is IVF-flat.
  private final ProductQuantizer pq;
  private final byte[][] pqCodes; // [n][M]; pqCodes[i] is the PQ code for vectors[i]

  private IvfIndex(
      BuoyIndex buoyIndex,
      ClusterPartition[] partitions,
      float[][] vectors,
      String[] ids,
      SimilarityFunction metric,
      ProductQuantizer pq,
      byte[][] pqCodes) {
    this.buoyIndex = buoyIndex;
    this.partitions = partitions;
    this.vectors = vectors;
    this.ids = ids;
    this.metric = metric;
    this.pq = pq;
    this.pqCodes = pqCodes;
  }

  /**
   * Builds an IVF index from {@code vectors} using {@code params}. When {@link
   * IvfBuildParams#pqEnabled()} is {@code true}, trains a {@link ProductQuantizer} on per-cluster
   * residuals and encodes each vector into an {@code M}-byte code; otherwise builds IVF-flat.
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

    ProductQuantizer pq = null;
    byte[][] pqCodes = null;
    if (params.pqEnabled()) {
      // Residuals r[i] = vectors[i] - centroid(assignments[i]). Training happens on the residuals
      // with center=false so buildADCTable() does not subtract any global centroid at query time
      // — the residual query (q - centroid(c)) is already the correct input per probed cluster.
      int dim = vectors[0].length;
      float[][] residuals = new float[n][dim];
      for (int i = 0; i < n; i++) {
        float[] centroid = buoys[assignments[i]];
        float[] v = vectors[i];
        float[] r = residuals[i];
        for (int d = 0; d < dim; d++) r[d] = v[d] - centroid[d];
      }
      pq =
          ProductQuantizer.train(
              new ArrayVectorDataset(residuals),
              params.pqSubspaces(),
              params.pqClusters(),
              false,
              1,
              params.pqAnisotropicThreshold());
      pqCodes = new byte[n][];
      for (int i = 0; i < n; i++) pqCodes[i] = pq.encode(residuals[i]);
    }

    return new IvfIndex(buoy, partitions, vectors, ids, metric, pq, pqCodes);
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
    if (pq != null) return pqSearch(request);
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

  /** Returns the {@link ProductQuantizer} used by this index, or {@code null} for IVF-flat. */
  public ProductQuantizer quantizer() {
    return pq;
  }

  /** Returns {@code true} if this index was built with product quantization enabled. */
  public boolean isQuantized() {
    return pq != null;
  }

  /**
   * Returns the per-ordinal PQ codes ({@code byte[n][M]}), or {@code null} for IVF-flat. Exposed
   * for codec serialisers; callers must not mutate the returned array.
   */
  public byte[][] pqCodes() {
    return pqCodes;
  }

  /**
   * Constructs an {@link IvfIndex} from pre-trained components — used by codec deserialisers to
   * avoid re-running KMeans on load. Builds an IVF-flat index.
   */
  public static IvfIndex fromPrebuilt(
      BuoyIndex buoyIndex,
      ClusterPartition[] partitions,
      float[][] vectors,
      SimilarityFunction metric) {
    return new IvfIndex(buoyIndex, partitions, vectors, null, metric, null, null);
  }

  /**
   * Constructs an {@link IvfIndex} from pre-trained components including a pre-trained {@link
   * ProductQuantizer} and its per-ordinal codes. Used by codec deserialisers for IVF-PQ.
   */
  public static IvfIndex fromPrebuilt(
      BuoyIndex buoyIndex,
      ClusterPartition[] partitions,
      float[][] vectors,
      SimilarityFunction metric,
      ProductQuantizer pq,
      byte[][] pqCodes) {
    return new IvfIndex(buoyIndex, partitions, vectors, null, metric, pq, pqCodes);
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
    return new IvfIndex(buoyIndex, partitions, vectors, null, metric, null, null);
  }

  @Override
  public void close() {
    // no off-heap resources in Phase 1
  }

  // --- IVF-PQ search ---

  /**
   * ADC search over PQ codes with per-cluster residuals. For each probed cluster {@code c}, a fresh
   * ADC lookup table is built from the residual query {@code q - centroid(c)}, and each ordinal in
   * the partition is scored via {@code M} table lookups. Candidates populate a wide heap sized
   * {@code k * rescoreFactor}; when {@code rescoreFactor > 1} the wide-heap contents are re-ranked
   * against full-precision vectors to produce the final top-k.
   *
   * <p>Score conventions (higher = more similar, consistent with IVF-flat):
   *
   * <ul>
   *   <li>EUCLIDEAN: {@code -adcSqL2(q - centroid(c), code)}
   *   <li>DOT_PRODUCT / MAXIMUM_INNER_PRODUCT: {@code q·centroid(c) + adcDot(q, code)}
   *   <li>COSINE: {@code adcDot(q, code)} as an unnormalised dot proxy. Use {@code rescoreFactor >=
   *       2} for correct cosine scores (the final rescore pass computes {@code VectorUtil.cosine}).
   * </ul>
   *
   * <p>HARMONY pruning and triangle-inequality early termination are both disabled on this path;
   * ADC scoring is already cheap enough that the scan is typically bounded by the product-quantised
   * table lookups, not by vector fetches.
   */
  private IvfSearchResult pqSearch(IvfSearchRequest request) {
    float[] query = request.query();
    int k = request.k();
    int rescoreFactor = request.rescoreFactor();
    int wideK = Math.min(Math.multiplyExact(k, rescoreFactor), vectors.length);
    if (wideK < 1) wideK = 1;
    int nprobe = Math.min(request.nprobe(), partitions.length);
    var route = buoyIndex.routeWithDistances(query, nprobe, request.gamma());
    int[] clusterIds = route.clusterIds();

    TopKHeap wide = new TopKHeap(wideK);
    boolean euclidean = metric == SimilarityFunction.EUCLIDEAN;
    boolean useDotTable = !euclidean; // dot/mip/cosine → dot ADC; L2 → sqL2 ADC
    int dim = query.length;
    int numSubspaces = pq.numSubspaces();

    int clustersScanned = 0;
    for (int i = 0; i < clusterIds.length; i++) {
      int cid = clusterIds[i];
      ClusterPartition partition = partitions[cid];
      int[] ords = partition.ordinals();
      if (ords.length == 0) {
        clustersScanned++;
        continue;
      }
      float[] centroid = partition.centroid();

      float[][] table;
      float baseScore; // per-cluster additive shift so scores are comparable across clusters
      if (euclidean) {
        float[] residualQuery = new float[dim];
        for (int d = 0; d < dim; d++) residualQuery[d] = query[d] - centroid[d];
        table = pq.buildADCTable(residualQuery, false);
        baseScore = 0f; // score = -sum(table[m][code[m]])
      } else {
        // dot-style metrics: ADC over raw query gives q·v_residual; add q·centroid(c) for the
        // full-vector dot. Cosine uses the same dot proxy (no normalisation on this path).
        table = pq.buildADCTable(query, true);
        baseScore =
            metric == SimilarityFunction.COSINE ? 0f : VectorUtil.dotProduct(query, centroid);
      }

      for (int ord : ords) {
        byte[] code = pqCodes[ord];
        float sum = 0f;
        for (int m = 0; m < numSubspaces; m++) sum += table[m][code[m] & 0xFF];
        float score = euclidean ? -sum : baseScore + sum;
        wide.offer(ord, score);
      }
      clustersScanned++;
    }

    // Rescore pass against full-precision vectors when requested.
    float minScore = request.minScore();
    TopKHeap.DrainResult wideDrain = wide.drainDescending();
    int[] wideOrds = wideDrain.ordinals();
    List<IvfHit> hits;
    if (rescoreFactor == 1) {
      hits = new ArrayList<>(wideOrds.length);
      float[] wideScores = wideDrain.scores();
      for (int i = 0; i < wideOrds.length; i++) {
        if (wideScores[i] < minScore) continue;
        int ord = wideOrds[i];
        hits.add(new IvfHit(ord, ids != null ? ids[ord] : null, wideScores[i]));
      }
    } else {
      TopKHeap finalHeap = new TopKHeap(k);
      for (int ord : wideOrds) {
        float score = score(query, vectors[ord]);
        if (score < minScore) continue;
        finalHeap.offer(ord, score);
      }
      TopKHeap.DrainResult drained = finalHeap.drainDescending();
      hits = new ArrayList<>(drained.ordinals().length);
      for (int i = 0; i < drained.ordinals().length; i++) {
        int ord = drained.ordinals()[i];
        hits.add(new IvfHit(ord, ids != null ? ids[ord] : null, drained.scores()[i]));
      }
    }
    return new IvfSearchResult(hits, clustersScanned);
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
