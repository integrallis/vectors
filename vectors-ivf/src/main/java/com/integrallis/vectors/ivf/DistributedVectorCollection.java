/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.ivf;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.storage.manifest.ManifestConflictException;
import com.integrallis.vectors.storage.manifest.ManifestGenerationPublisher.GenerationAnnouncer;
import com.integrallis.vectors.storage.manifest.ManifestStore;
import com.integrallis.vectors.storage.manifest.StorageManifest;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

  private static final Logger LOG = LoggerFactory.getLogger(DistributedVectorCollection.class);

  // ─── WAL record tags ─────────────────────────────────────────────────────
  private static final byte TAG_ADD = 1;
  private static final byte TAG_COMMIT = 4;

  /** T3 key under which the encoded {@link BuoyIndex} is persisted for WAL replay. */
  private static final String ROUTING_KEY = "_routing-index";

  /**
   * Object-storage key of the durable generation manifest. The manifest is a CAS-published pointer
   * (via {@link ManifestStore}) that makes the committed generation <em>discoverable</em> to remote
   * readers — the WAL is local-only — and monotonic across writers. It is published <em>after</em>
   * the WAL COMMIT, so the WAL remains the authority for local crash recovery while the manifest
   * gives DartVault an object-storage-native, race-safe generation pointer (the same shape
   * TurboPuffer/Lance/Delta use).
   */
  private static final String MANIFEST_KEY = "_manifest";

  private static final String MANIFEST_COLLECTION_ENTRY = "collection";
  private static final int MAX_MANIFEST_ATTEMPTS = 16;
  private static final String DEFAULT_WRITER = "dartvault";

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

  // Manifest-publish configuration (opt-in gossip + multi-writer provenance). Volatile: set by the
  // owning node after build()/open(), read on the commit path.
  private volatile GenerationAnnouncer generationAnnouncer = GenerationAnnouncer.NONE;
  private volatile String writerId = DEFAULT_WRITER;

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
   * Wires an announcer that receives {@code (generation, contentHash)} after every committed
   * manifest publish — bridge it to {@code GossipClusterMembership.announceVersion(long, String)}
   * so a commit tells followers to reload, guarded by the gossip max-wins monotonicity check.
   * Defaults to a no-op (single-node / no gossip).
   */
  public void setGenerationAnnouncer(GenerationAnnouncer announcer) {
    this.generationAnnouncer = java.util.Objects.requireNonNull(announcer, "announcer");
  }

  /** Sets the writer id recorded as manifest provenance (distinct per writer for multi-writer). */
  public void setWriterId(String writerId) {
    this.writerId = java.util.Objects.requireNonNull(writerId, "writerId");
  }

  /**
   * Publishes {@code generation} to the durable object-storage manifest via a monotonic CAS, then
   * announces it. If the manifest is already at or past {@code generation} (a concurrent writer, or
   * an idempotent re-publish) it does not regress. Called <em>after</em> the WAL COMMIT, so the WAL
   * stays authoritative for local recovery; the manifest is the discoverable, race-safe pointer for
   * remote readers and the gossip generation source.
   */
  private static void publishManifest(
      StorageBackend backend,
      String writerId,
      GenerationAnnouncer announcer,
      long generation,
      int vectorCount,
      long committedAtEpochMs)
      throws IOException {
    ManifestStore store = new ManifestStore(backend, MANIFEST_KEY);
    for (int attempt = 0; attempt < MAX_MANIFEST_ATTEMPTS; attempt++) {
      ManifestStore.Loaded loaded = store.load();
      StorageManifest current = loaded.manifest();
      if (!current.isEmpty() && current.generation() >= generation) {
        return; // already at/past this generation — idempotent, never regress
      }
      StorageManifest next =
          new StorageManifest(
              generation,
              contentHash(generation, vectorCount),
              Map.of(MANIFEST_COLLECTION_ENTRY, generation),
              committedAtEpochMs,
              writerId);
      if (store.compareAndPut(next, loaded.etag()) != null) {
        announcer.announce(generation, next.contentHash());
        return;
      }
      // CAS lost to a concurrent writer — retry; the guard above returns if they reached >= us.
    }
    throw new ManifestConflictException(
        "manifest publish for generation "
            + generation
            + " lost after "
            + MAX_MANIFEST_ATTEMPTS
            + " attempts");
  }

  /** Cheap, deterministic fingerprint of the committed state for the manifest content hash. */
  private static String contentHash(long generation, int vectorCount) {
    return Integer.toHexString(31 * vectorCount + Long.hashCode(generation));
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

    // Publish the durable generation-0 manifest so a remote reader can discover the collection.
    publishManifest(
        t3Backend, DEFAULT_WRITER, GenerationAnnouncer.NONE, 0L, n, System.currentTimeMillis());

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

    // 5. Resolve the live generation. The local WAL is authoritative for recovery, but a durable
    // manifest may be ahead of (or the only source for) a reader with no local WAL — take the max
    // so
    // a remote/replica reader discovers the published generation. Absent manifest → WAL only.
    long resolvedGeneration = gen[0];
    StorageManifest manifest = new ManifestStore(t3Backend, MANIFEST_KEY).load().manifest();
    if (!manifest.isEmpty()) {
      resolvedGeneration = Math.max(resolvedGeneration, manifest.generation());
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
        resolvedGeneration);
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
   * Discards all staged-but-uncommitted vectors (the buffer that {@link #add} appends to and {@link
   * #commit} consumes). Nothing durable is affected — staged vectors were never written to the WAL
   * or the tiered clusters. This lets a caller that re-runs a partially-applied {@code add()} loop
   * (e.g. a retried batch stage) start from a clean slate instead of double-staging the docs it had
   * already added on the failed attempt.
   */
  public void discardStaging() {
    rwLock.writeLock().lock();
    try {
      staging.clear();
      stagingIds.clear();
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
      publishManifest(
          t3Backend,
          writerId,
          generationAnnouncer,
          generation,
          allVectors.size(),
          System.currentTimeMillis());
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
    publishManifest(
        t3Backend,
        writerId,
        generationAnnouncer,
        generation,
        allVectors.size(),
        System.currentTimeMillis());
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
      if (k <= 0) return List.of();
      int[] clusterIds = routingIndex.route(query, Math.min(nprobe, this.k), 0f);

      // Merge cluster + staging candidates through a bounded size-k min-heap instead of collecting
      // every candidate into a list and full-sorting it. Ordinals are kept raw here (cluster hits
      // are global ordinals >= 0; staging hits use -(i+1)); the document id is re-attached only for
      // the surviving top-k when the heap is drained. Ties at the k-boundary may resolve to a
      // different equally-scored ordinal than the previous stable full-sort.
      TopKHeap heap = new TopKHeap(k);
      for (int cid : clusterIds) {
        clusters[cid].recordAccess();
        for (IvfHit hit : clusters[cid].scan(query, k, -Float.MAX_VALUE)) {
          heap.offer(hit.ordinal(), hit.score());
        }
      }

      // Scan staging exactly
      for (int i = 0; i < staging.size(); i++) {
        heap.offer(-(i + 1), score(query, staging.get(i)));
      }

      TopKHeap.DrainResult drained = heap.drainDescending(); // descending score
      int[] ords = drained.ordinals();
      float[] scores = drained.scores();
      List<IvfHit> results = new ArrayList<>(ords.length);
      for (int i = 0; i < ords.length; i++) {
        int ord = ords[i];
        String id;
        if (ord < 0) {
          id = stagingIds.get(-ord - 1); // staging entry
        } else {
          id = ord < allIds.size() ? allIds.get(ord) : null; // committed entry
        }
        results.add(new IvfHit(ord, id, scores[i]));
      }
      return results;
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

  /**
   * Returns the document id at ordinal {@code i} in stable iteration order, where {@code
   * 0..allVectors.size()-1} are committed entries and the tail is the staging buffer appended in
   * insertion order.
   */
  public String idAt(int i) {
    rwLock.readLock().lock();
    try {
      int committed = allIds.size();
      if (i < 0 || i >= committed + stagingIds.size()) {
        throw new IndexOutOfBoundsException(
            "idAt: " + i + " of " + (committed + stagingIds.size()));
      }
      return i < committed ? allIds.get(i) : stagingIds.get(i - committed);
    } finally {
      rwLock.readLock().unlock();
    }
  }

  /** Returns a defensive copy of the vector at ordinal {@code i} (committed or staged). */
  public float[] vectorAt(int i) {
    rwLock.readLock().lock();
    try {
      int committed = allVectors.size();
      if (i < 0 || i >= committed + staging.size()) {
        throw new IndexOutOfBoundsException(
            "vectorAt: " + i + " of " + (committed + staging.size()));
      }
      float[] src = i < committed ? allVectors.get(i) : staging.get(i - committed);
      return Arrays.copyOf(src, src.length);
    } finally {
      rwLock.readLock().unlock();
    }
  }

  /**
   * Returns an immutable snapshot of every live document id in stable ordinal order: committed ids
   * first, staging ids appended. The list does not reflect later mutations.
   */
  public List<String> allIdsView() {
    rwLock.readLock().lock();
    try {
      List<String> out = new ArrayList<>(allIds.size() + stagingIds.size());
      out.addAll(allIds);
      out.addAll(stagingIds);
      return List.copyOf(out);
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

  /**
   * Enables object-storage read-through on every cluster (P1.8): subsequent {@link #search} calls
   * fetch each probed cluster's vectors from the T3 {@link StorageBackend} on first probe and cache
   * them, instead of scanning the heap-resident global array — making per-query cold/warm
   * object-storage reads observable. Off by default; existing callers and the R2 integration tests
   * are unaffected. Call after ingest/commits, since {@link #commit} rebuilds dirty clusters
   * without read-through.
   */
  public void enableReadThrough() {
    rwLock.writeLock().lock();
    try {
      for (TieredCluster c : clusters) {
        c.enableReadThrough(t3Backend);
      }
    } finally {
      rwLock.writeLock().unlock();
    }
  }

  /**
   * Drops every cluster's read-through cache so the next probe re-fetches from T3 (forces cold).
   */
  public void dropReadThroughCaches() {
    rwLock.writeLock().lock();
    try {
      for (TieredCluster c : clusters) {
        c.dropReadThroughCache();
      }
    } finally {
      rwLock.writeLock().unlock();
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

  /**
   * Seals the active WAL segment and starts a new one. Bulk ingestors call this between large
   * batches to bound segment file size and to make crash-recovery {@code replay()} cost
   * predictable. No-ops when the active segment is empty.
   */
  public void rotateWalSegment() throws IOException {
    rwLock.writeLock().lock();
    try {
      wal.seal();
    } finally {
      rwLock.writeLock().unlock();
    }
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
        // Tier promotion/demotion failure must not break commit, but it must be visible: a cluster
        // that silently fails to persist/demote is otherwise invisible to operators.
        LOG.warn("tier eviction failed for cluster {} (continuing commit)", c, e);
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
