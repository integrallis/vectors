package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A single IVF cluster with tiered storage: T1 (SQ8 heap), T2/T3 (float32 via {@link
 * StorageBackend}).
 *
 * <p>T0 (1-bit RaBitQ, globally replicated) is managed by the owning {@link BuoyIndex} and is
 * intentionally not stored here — coarse routing happens above this class.
 *
 * <p>Tier lifecycle:
 *
 * <ol>
 *   <li>Cold start: vectors live in T3 (written via {@link #storeT3}).
 *   <li>Warm: after {@link TierPolicy} promotes to T2, the caller stores to a local {@link
 *       StorageBackend}. No additional state is maintained here for T2.
 *   <li>Hot: caller invokes {@link #materializeT1()} — SQ8 bytes land in heap.
 *   <li>Eviction: caller invokes {@link #evictT1()} — heap bytes are released.
 * </ol>
 *
 * <p>Search: {@link #scan(float[], int, float)} automatically uses T1 (SQ8) if materialised;
 * otherwise falls back to exact distance computation against {@code globalVectors}.
 */
public final class TieredCluster {

  private static final int HEADER_BYTES = 8; // n(4) + dim(4)

  private final ClusterPartition partition;
  private final float[][] globalVectors;
  private final SimilarityFunction metric;
  private final AtomicInteger accessCount = new AtomicInteger(0);

  /** T1: SQ8-encoded representation; null when not materialised. */
  private volatile CompressedVectors t1Data;

  /**
   * Constructs a {@code TieredCluster} from an existing partition and the global vector array.
   *
   * @param partition the logical cluster (ordinals + centroid)
   * @param globalVectors full-precision dataset; {@code globalVectors[partition.ordinals()[i]]}
   *     gives the i-th vector in this cluster
   * @param metric distance metric used for scoring
   */
  public TieredCluster(
      ClusterPartition partition, float[][] globalVectors, SimilarityFunction metric) {
    this.partition = partition;
    this.globalVectors = globalVectors;
    this.metric = metric;
  }

  // ─── access tracking ──────────────────────────────────────────────────────

  /** Increments the access counter by 1 (thread-safe). */
  public void recordAccess() {
    accessCount.incrementAndGet();
  }

  /** Returns the total number of times this cluster has been probed. */
  public int accessCount() {
    return accessCount.get();
  }

  // ─── T1 materialization ───────────────────────────────────────────────────

  /** Returns {@code true} if T1 (SQ8) data is currently materialised in heap. */
  public boolean hasT1() {
    return t1Data != null;
  }

  /**
   * Materialises the T1 (SQ8) tier by training a {@link ScalarQuantizer} on this cluster's vectors
   * and encoding them. Subsequent {@link #scan} calls will use this representation.
   */
  public void materializeT1() {
    float[][] clusterVecs = extractClusterVectors();
    ArrayVectorDataset dataset = new ArrayVectorDataset(clusterVecs);
    ScalarQuantizer sq = ScalarQuantizer.train(dataset);
    this.t1Data = sq.encodeAll(dataset);
  }

  /** Releases T1 heap data, returning to exact-scan mode. */
  public void evictT1() {
    this.t1Data = null;
  }

  // ─── T3 storage ───────────────────────────────────────────────────────────

  /**
   * Serialises this cluster's vectors to the given {@link StorageBackend} under the key {@code
   * "cluster-<clusterId>"}.
   */
  public void storeT3(StorageBackend backend) throws IOException {
    float[][] vecs = extractClusterVectors();
    backend.put(t3Key(), serializeVectors(vecs));
  }

  /**
   * Loads and deserialises this cluster's vectors from the given {@link StorageBackend}.
   *
   * @return the cluster's full-precision float32 vectors
   */
  public float[][] fetchT3(StorageBackend backend) throws IOException {
    return deserializeVectors(backend.get(t3Key()));
  }

  // ─── search ───────────────────────────────────────────────────────────────

  /**
   * Scans this cluster and returns the top-{@code k} hits scored against {@code query}.
   *
   * <p>If T1 data is materialised, uses SQ8 approximate scoring; otherwise uses exact distances
   * against {@code globalVectors}.
   *
   * @param query the query vector
   * @param k maximum number of hits to return
   * @param minScore minimum score threshold (hits below are discarded)
   * @return hits in descending score order, size ≤ k
   */
  public List<IvfHit> scan(float[] query, int k, float minScore) {
    return (t1Data != null) ? scanWithT1(query, k, minScore) : scanExact(query, k, minScore);
  }

  // ─── internals ─────────────────────────────────────────────────────────────

  private String t3Key() {
    return "cluster-" + partition.clusterId();
  }

  private float[][] extractClusterVectors() {
    int[] ordinals = partition.ordinals();
    float[][] vecs = new float[ordinals.length][];
    for (int i = 0; i < ordinals.length; i++) vecs[i] = globalVectors[ordinals[i]];
    return vecs;
  }

  private List<IvfHit> scanExact(float[] query, int k, float minScore) {
    int[] ordinals = partition.ordinals();
    List<IvfHit> hits = new ArrayList<>(Math.min(k, ordinals.length));
    for (int global : ordinals) {
      float s = score(query, globalVectors[global]);
      if (s >= minScore) hits.add(new IvfHit(global, null, s));
    }
    // IvfHit.compareTo is descending (higher score first) — natural order gives top-k first
    hits.sort(null);
    return hits.size() <= k ? hits : new ArrayList<>(hits.subList(0, k));
  }

  private List<IvfHit> scanWithT1(float[] query, int k, float minScore) {
    int[] ordinals = partition.ordinals();
    var scoreFunction = t1Data.scoreFunctionFor(query, metric);
    List<IvfHit> hits = new ArrayList<>(Math.min(k, ordinals.length));
    for (int local = 0; local < ordinals.length; local++) {
      float s = scoreFunction.score(local);
      if (s >= minScore) hits.add(new IvfHit(ordinals[local], null, s));
    }
    hits.sort(null);
    return hits.size() <= k ? hits : new ArrayList<>(hits.subList(0, k));
  }

  private float score(float[] a, float[] b) {
    return switch (metric) {
      case EUCLIDEAN -> -VectorUtil.squareDistance(a, b);
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> VectorUtil.dotProduct(a, b);
      case COSINE -> VectorUtil.cosine(a, b);
    };
  }

  private static byte[] serializeVectors(float[][] vecs) {
    int n = vecs.length;
    int dim = n > 0 ? vecs[0].length : 0;
    ByteBuffer buf = ByteBuffer.allocate(HEADER_BYTES + n * dim * Float.BYTES);
    buf.putInt(n);
    buf.putInt(dim);
    for (float[] v : vecs) for (float f : v) buf.putFloat(f);
    return buf.array();
  }

  private static float[][] deserializeVectors(byte[] bytes) {
    ByteBuffer buf = ByteBuffer.wrap(bytes);
    int n = buf.getInt();
    int dim = buf.getInt();
    float[][] vecs = new float[n][dim];
    for (float[] v : vecs) for (int d = 0; d < dim; d++) v[d] = buf.getFloat();
    return vecs;
  }
}
