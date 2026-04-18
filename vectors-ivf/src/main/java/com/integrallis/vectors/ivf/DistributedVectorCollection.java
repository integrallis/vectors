package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.storage.wal.SegmentedWriteAheadLog;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A WAL-durable, multi-cluster vector collection that coordinates {@link TieredCluster}s routed by
 * a {@link BuoyIndex}, with automatic tier promotion via {@link TierPolicy}.
 *
 * <p><b>Write path</b>: {@link #add} stages a vector in memory; {@link #commit} assigns staged
 * vectors to nearest clusters, writes WAL records (ADD × N + COMMIT), rebuilds {@link
 * TieredCluster}s with the updated global vector array, stores T3 snapshots, and calls {@link
 * #applyTierPolicy}.
 *
 * <p><b>Read path</b>: {@link #search} routes the query to {@code nprobe} clusters via the {@link
 * BuoyIndex}, scans each {@link TieredCluster} (T1 or exact), scans the staging buffer exactly, and
 * returns the global top-k result.
 *
 * <p><b>Durability</b>: every {@link #commit} appends a self-contained WAL batch (ADD* + COMMIT) to
 * a {@link SegmentedWriteAheadLog}. Uncommitted staged vectors are intentionally volatile — only
 * committed state survives a crash.
 */
public final class DistributedVectorCollection implements AutoCloseable {

  // ─── WAL record tags ─────────────────────────────────────────────────────
  private static final byte TAG_ADD = 1;
  private static final byte TAG_COMMIT = 4;

  /** T3 key under which the encoded {@link BuoyIndex} is persisted for WAL replay. */
  private static final String ROUTING_KEY = "_routing-index";

  // ─── routing ─────────────────────────────────────────────────────────────
  private final BuoyIndex routingIndex;
  private final int k;
  private final SimilarityFunction metric;

  // ─── global state (committed) ────────────────────────────────────────────
  private final List<float[]> allVectors;
  private final List<String> allIds;
  private final List<List<Integer>> clusterOrdinals; // per cluster
  private TieredCluster[] clusters;

  // ─── staging (uncommitted) ───────────────────────────────────────────────
  private final List<float[]> staging = new ArrayList<>();
  private final List<String> stagingIds = new ArrayList<>();

  // ─── infrastructure ──────────────────────────────────────────────────────
  private final TierPolicy tierPolicy;
  private final StorageBackend t3Backend;
  private final SegmentedWriteAheadLog wal;
  private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
  private long generation;

  private DistributedVectorCollection(
      BuoyIndex routingIndex,
      int k,
      List<float[]> allVectors,
      List<String> allIds,
      List<List<Integer>> clusterOrdinals,
      TieredCluster[] clusters,
      TierPolicy tierPolicy,
      SimilarityFunction metric,
      StorageBackend t3Backend,
      SegmentedWriteAheadLog wal,
      long generation) {
    this.routingIndex = routingIndex;
    this.k = k;
    this.allVectors = allVectors;
    this.allIds = allIds;
    this.clusterOrdinals = clusterOrdinals;
    this.clusters = clusters;
    this.tierPolicy = tierPolicy;
    this.metric = metric;
    this.t3Backend = t3Backend;
    this.wal = wal;
    this.generation = generation;
  }

  /**
   * Builds a new {@code DistributedVectorCollection} from an initial vector dataset.
   *
   * <p>Trains the routing {@link BuoyIndex}, assigns initial vectors to clusters, writes T3
   * snapshots, writes the initial COMMIT record to the WAL, and returns the collection.
   */
  public static DistributedVectorCollection build(
      float[][] vectors,
      String[] ids,
      SimilarityFunction metric,
      IvfBuildParams params,
      ClusterSplitter splitter,
      TierPolicy tierPolicy,
      Path walDir,
      StorageBackend t3Backend)
      throws IOException {

    int n = vectors.length;
    int k = Math.min(params.k(), n);

    // Train routing index
    BuoyIndex routingIndex = BuoyIndex.train(vectors, k, metric, params.buildSoar(), params.seed());

    // Assign initial vectors to clusters
    List<List<Integer>> clusterOrdinals = new ArrayList<>(k);
    for (int c = 0; c < k; c++) clusterOrdinals.add(new ArrayList<>());
    for (int i = 0; i < n; i++) {
      int cid = routingIndex.route(vectors[i], 1, 0f)[0];
      clusterOrdinals.get(cid).add(i);
    }

    // Build TieredClusters and store T3 snapshots
    float[][] centroids = routingIndex.buoyVectors();
    TieredCluster[] clusters = new TieredCluster[k];
    for (int c = 0; c < k; c++) {
      int[] ords = clusterOrdinals.get(c).stream().mapToInt(Integer::intValue).toArray();
      clusters[c] = new TieredCluster(ClusterPartition.of(c, centroids[c], ords), vectors, metric);
      clusters[c].storeT3(t3Backend);
    }

    // Persist routing index to T3 so open() can replay without retraining
    t3Backend.put(ROUTING_KEY, routingIndex.encode());

    List<float[]> allVectors = new ArrayList<>(Arrays.asList(vectors));
    List<String> allIds = (ids != null) ? new ArrayList<>(Arrays.asList(ids)) : new ArrayList<>();

    // WAL: write ADD records for all initial vectors so open() can fully replay
    SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(walDir);
    for (int i = 0; i < n; i++) {
      String docId = (ids != null) ? ids[i] : "doc-" + i;
      wal.append(encodeAdd(docId, vectors[i]));
    }
    wal.append(encodeCommit(0L));

    return new DistributedVectorCollection(
        routingIndex,
        k,
        allVectors,
        allIds,
        clusterOrdinals,
        clusters,
        tierPolicy,
        metric,
        t3Backend,
        wal,
        0L);
  }

  /**
   * Reopens an existing collection by loading the {@link BuoyIndex} from {@code t3Backend} and
   * replaying all sealed WAL segments in {@code walDir} to reconstruct cluster assignments.
   *
   * <p>Only committed state is restored — any staged (uncommitted) vectors from a previous session
   * are treated as lost (intentionally volatile).
   */
  public static DistributedVectorCollection open(
      Path walDir, SimilarityFunction metric, TierPolicy tierPolicy, StorageBackend t3Backend)
      throws IOException {

    // 1. Restore routing index from T3
    byte[] routingBytes = t3Backend.get(ROUTING_KEY);
    if (routingBytes == null)
      throw new IOException("routing index not found in T3 backend (key=" + ROUTING_KEY + ")");
    BuoyIndex routingIndex = BuoyIndex.decode(routingBytes);
    int k = routingIndex.k();
    float[][] centroids = routingIndex.buoyVectors();

    // 2. Initialise replay buffers
    List<float[]> allVectors = new ArrayList<>();
    List<String> allIds = new ArrayList<>();
    List<List<Integer>> clusterOrdinals = new ArrayList<>(k);
    for (int c = 0; c < k; c++) clusterOrdinals.add(new ArrayList<>());
    long[] gen = {0};

    // 3. Open WAL, seal current (empty) segment, replay existing sealed segments
    SegmentedWriteAheadLog wal = new SegmentedWriteAheadLog(walDir);
    wal.seal();
    wal.replay(
        entry -> {
          if (entry.length == 0) return;
          byte tag = entry[0];
          if (tag == TAG_ADD) {
            ByteBuffer buf = ByteBuffer.wrap(entry, 1, entry.length - 1);
            short idLen = buf.getShort();
            byte[] idBytes = new byte[idLen];
            buf.get(idBytes);
            String id = new String(idBytes, StandardCharsets.UTF_8);
            int dim = buf.getInt();
            float[] vec = new float[dim];
            for (int i = 0; i < dim; i++) vec[i] = buf.getFloat();
            int ordinal = allVectors.size();
            allVectors.add(vec);
            allIds.add(id);
            int cid = routingIndex.route(vec, 1, 0f)[0];
            clusterOrdinals.get(cid).add(ordinal);
          } else if (tag == TAG_COMMIT) {
            gen[0] = ByteBuffer.wrap(entry, 1, 8).getLong();
          }
        });

    // 4. Reconstruct TieredClusters from replayed state
    float[][] vecArray = allVectors.toArray(new float[0][]);
    TieredCluster[] clusters = new TieredCluster[k];
    for (int c = 0; c < k; c++) {
      int[] ords = clusterOrdinals.get(c).stream().mapToInt(Integer::intValue).toArray();
      clusters[c] = new TieredCluster(ClusterPartition.of(c, centroids[c], ords), vecArray, metric);
    }

    return new DistributedVectorCollection(
        routingIndex,
        k,
        allVectors,
        allIds,
        clusterOrdinals,
        clusters,
        tierPolicy,
        metric,
        t3Backend,
        wal,
        gen[0]);
  }

  // ─── write path ──────────────────────────────────────────────────────────

  /** Stages {@code vector} with the given {@code id}. Not durable until {@link #commit}. */
  public void add(String id, float[] vector) {
    rwLock.writeLock().lock();
    try {
      staging.add(vector);
      stagingIds.add(id);
    } finally {
      rwLock.writeLock().unlock();
    }
  }

  /**
   * Commits all staged vectors: assigns to nearest cluster, writes WAL (ADD* + COMMIT), rebuilds
   * {@link TieredCluster}s, updates T3, and applies the {@link TierPolicy}.
   */
  public void commit() throws IOException {
    rwLock.writeLock().lock();
    try {
      commitUnderLock();
    } finally {
      rwLock.writeLock().unlock();
    }
  }

  private void commitUnderLock() throws IOException {
    if (staging.isEmpty()) {
      generation++;
      wal.append(encodeCommit(generation));
      applyTierPolicy();
      return;
    }

    // Assign staged vectors to clusters, track which clusters received new vectors
    boolean[] dirty = new boolean[k];
    for (int i = 0; i < staging.size(); i++) {
      float[] vec = staging.get(i);
      String id = stagingIds.get(i);
      int globalOrdinal = allVectors.size();
      allVectors.add(vec);
      allIds.add(id);
      int cid = routingIndex.route(vec, 1, 0f)[0];
      clusterOrdinals.get(cid).add(globalOrdinal);
      dirty[cid] = true;
      wal.append(encodeAdd(id, vec));
    }
    staging.clear();
    stagingIds.clear();

    // Rebuild only clusters that received new vectors
    float[][] newVecArray = allVectors.toArray(new float[0][]);
    float[][] centroids = routingIndex.buoyVectors();
    for (int c = 0; c < k; c++) {
      if (!dirty[c]) continue;
      int[] ords = clusterOrdinals.get(c).stream().mapToInt(Integer::intValue).toArray();
      boolean hadT1 = clusters[c].hasT1();
      int prevCount = clusters[c].accessCount();
      clusters[c] =
          new TieredCluster(
              ClusterPartition.of(c, centroids[c], ords), newVecArray, metric, prevCount);
      clusters[c].storeT3(t3Backend);
      if (hadT1) clusters[c].materializeT1();
    }

    generation++;
    wal.append(encodeCommit(generation));
    applyTierPolicy();
  }

  // ─── read path ───────────────────────────────────────────────────────────

  /**
   * Routes the query to {@code nprobe} clusters, scans each (T1 or exact), scans staged vectors,
   * and returns the global top-{@code k} hits in descending score order.
   *
   * <p>Uses a read lock, allowing concurrent searches while writes are exclusive.
   */
  public List<IvfHit> search(float[] query, int k, int nprobe) {
    rwLock.readLock().lock();
    try {
      int[] clusterIds = routingIndex.route(query, Math.min(nprobe, this.k), 0f);

      List<IvfHit> candidates = new ArrayList<>();
      for (int cid : clusterIds) {
        clusters[cid].recordAccess();
        for (IvfHit hit : clusters[cid].scan(query, k, -Float.MAX_VALUE)) {
          // Re-attach the document id from the global id list
          String id =
              hit.ordinal() >= 0 && hit.ordinal() < allIds.size()
                  ? allIds.get(hit.ordinal())
                  : null;
          candidates.add(new IvfHit(hit.ordinal(), id, hit.score()));
        }
      }

      // Scan staging exactly
      for (int i = 0; i < staging.size(); i++) {
        candidates.add(new IvfHit(-(i + 1), stagingIds.get(i), score(query, staging.get(i))));
      }

      candidates.sort(null); // IvfHit natural order = descending score
      return candidates.size() <= k ? candidates : new ArrayList<>(candidates.subList(0, k));
    } finally {
      rwLock.readLock().unlock();
    }
  }

  // ─── observability ───────────────────────────────────────────────────────

  /** Total vector count: committed + staged. */
  public int size() {
    rwLock.readLock().lock();
    try {
      return allVectors.size() + staging.size();
    } finally {
      rwLock.readLock().unlock();
    }
  }

  /** Number of staged (uncommitted) vectors. */
  public int stagingSize() {
    rwLock.readLock().lock();
    try {
      return staging.size();
    } finally {
      rwLock.readLock().unlock();
    }
  }

  /** Number of clusters currently materialised at T1. */
  public int t1ClusterCount() {
    rwLock.readLock().lock();
    try {
      int count = 0;
      for (TieredCluster c : clusters) if (c.hasT1()) count++;
      return count;
    } finally {
      rwLock.readLock().unlock();
    }
  }

  /** Current WAL generation (incremented on each commit). */
  public long generation() {
    return generation;
  }

  @Override
  public void close() throws IOException {
    wal.close();
  }

  // ─── tier management ─────────────────────────────────────────────────────

  private void applyTierPolicy() {
    Map<Integer, Integer> counts = new java.util.HashMap<>(k * 2);
    for (int c = 0; c < k; c++) counts.put(c, clusters[c].accessCount());
    Map<Integer, TierPolicy.Tier> tiers = tierPolicy.applyAll(counts);
    for (int c = 0; c < k; c++) {
      try {
        clusters[c].evictToTier(tiers.get(c), t3Backend);
      } catch (java.io.IOException e) {
        // Log and continue — tier promotion/demotion failure should not break commit
      }
    }
  }

  // ─── scoring ─────────────────────────────────────────────────────────────

  private float score(float[] a, float[] b) {
    return switch (metric) {
      case EUCLIDEAN -> -com.integrallis.vectors.core.VectorUtil.squareDistance(a, b);
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT ->
          com.integrallis.vectors.core.VectorUtil.dotProduct(a, b);
      case COSINE -> com.integrallis.vectors.core.VectorUtil.cosine(a, b);
    };
  }

  // ─── WAL codec ───────────────────────────────────────────────────────────

  private static byte[] encodeAdd(String id, float[] vector) {
    byte[] idBytes = id.getBytes(StandardCharsets.UTF_8);
    ByteBuffer buf = ByteBuffer.allocate(1 + 2 + idBytes.length + 4 + vector.length * 4);
    buf.put(TAG_ADD);
    buf.putShort((short) idBytes.length);
    buf.put(idBytes);
    buf.putInt(vector.length);
    for (float f : vector) buf.putFloat(f);
    return buf.array();
  }

  private static byte[] encodeCommit(long generation) {
    ByteBuffer buf = ByteBuffer.allocate(1 + 8);
    buf.put(TAG_COMMIT);
    buf.putLong(generation);
    return buf.array();
  }
}
