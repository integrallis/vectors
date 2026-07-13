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
import com.integrallis.vectors.core.VectorUtil;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
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
  private final AtomicInteger accessCount;

  /**
   * The generation this cluster's object-storage payloads are keyed under. Payload keys are
   * generation-scoped ({@code gen-<N>/cluster-<id>}) so a commit that rewrites only the dirty
   * clusters never overwrites the live generation's objects: the manifest CAS atomically switches
   * which per-cluster generation is current, and a partial write leaves the prior generation's
   * objects intact (crash-atomic, and time-travel readable). Set at store time and on open from the
   * manifest's per-cluster entry. Volatile: written under the collection write lock, read on the
   * lock-free read-through path.
   */
  private volatile long t3Generation;

  /** T1: SQ8-encoded representation; null when not materialised. */
  private volatile CompressedVectors t1Data;

  /**
   * Object-storage read-through state. Combining the backend and cache into a single immutable
   * snapshot lets every reader observe a consistent (backend, cache) pair without locking — the
   * previous two-volatile-field layout allowed a reader to observe a fresh backend with a stale
   * cache (or vice versa) while another thread was mid-{@link #enableReadThrough} / {@link
   * #dropReadThroughCache}. Null means read-through is disabled (the default).
   */
  private volatile ReadThroughState readThrough;

  /** Guards the cold-load section so concurrent probes single-flight one backend fetch. */
  private final Object readThroughLoadLock = new Object();

  private record ReadThroughState(StorageBackend backend, float[][] cache) {}

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
    this(partition, globalVectors, metric, 0);
  }

  /**
   * Constructs a {@code TieredCluster} with a pre-set access count (used when rebuilding a dirty
   * cluster during commit to preserve tier-policy state).
   *
   * @param partition the logical cluster (ordinals + centroid)
   * @param globalVectors full-precision dataset
   * @param metric distance metric used for scoring
   * @param initialAccessCount access count carried over from the previous generation
   */
  public TieredCluster(
      ClusterPartition partition,
      float[][] globalVectors,
      SimilarityFunction metric,
      int initialAccessCount) {
    this.partition = partition;
    this.globalVectors = globalVectors;
    this.metric = metric;
    this.accessCount = new AtomicInteger(initialAccessCount);
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

  /** The generation this cluster's object-storage payloads are keyed under. */
  public long t3Generation() {
    return t3Generation;
  }

  /**
   * Sets the generation for this cluster's object-storage payload keys. Called before {@link
   * #storeT3} at commit (the new generation) and on {@code open} from the manifest's per-cluster
   * entry, so reads and writes target the same generation-scoped key.
   */
  public void setT3Generation(long generation) {
    this.t3Generation = generation;
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

  // ─── HyperDoor ────────────────────────────────────────────────────────────

  /**
   * Returns a {@link HyperDoor} reflecting this cluster's current tier state.
   *
   * <ul>
   *   <li>{@code clusterOrdinal} = cluster id from the partition.
   *   <li>{@code t0BitOffset} = 0 (T0 is managed at the {@link BuoyIndex} level).
   *   <li>{@code t1ByteOffset} = 0 if T1 is materialised, -1 otherwise.
   *   <li>{@code t2FileOffset} = -1 (T2 mmap not tracked in this implementation).
   *   <li>{@code t3ObjectOffset} = 0 (T3 always persisted under the cluster's T3 key).
   * </ul>
   */
  public HyperDoor hyperDoor() {
    return new HyperDoor(
        partition.clusterId(),
        0L, // T0: managed by BuoyIndex
        hasT1() ? 0 : -1, // T1: heap-resident flag
        -1L, // T2: mmap not tracked
        0L // T3: always available in backend
        );
  }

  // ─── tier eviction ────────────────────────────────────────────────────────

  /**
   * Demotes this cluster's in-memory representation to {@code target} tier, persisting data to
   * {@code backend} as required and releasing in-heap resources that are no longer needed.
   *
   * <ul>
   *   <li>{@link TierPolicy.Tier#T3}: persist float32 vectors under the T3 key; evict T1.
   *   <li>{@link TierPolicy.Tier#T2}: persist float32 vectors under the T2 key; evict T1.
   *   <li>{@link TierPolicy.Tier#T1}: materialise T1 (SQ8) if not already present.
   *   <li>{@link TierPolicy.Tier#T0}: no-op (T0 is managed by the owning {@link BuoyIndex}).
   * </ul>
   *
   * @param target desired storage tier after eviction
   * @param backend the {@link StorageBackend} used for T2/T3 persistence
   * @throws IOException if persistence to the backend fails
   */
  public void evictToTier(TierPolicy.Tier target, StorageBackend backend) throws IOException {
    switch (target) {
      case T3 -> {
        storeT3(backend);
        evictT1();
      }
      case T2 -> {
        storeT2(backend);
        evictT1();
      }
      case T1 -> {
        if (!hasT1()) materializeT1();
      }
      case T0 -> {
        // T0 is managed by BuoyIndex; no action required here.
      }
    }
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
    if (k <= 0) return List.of();
    ReadThroughState rt = readThrough;
    if (rt != null) {
      return scanReadThrough(query, k, minScore, rt);
    }
    return (t1Data != null) ? scanWithT1(query, k, minScore) : scanExact(query, k, minScore);
  }

  // ─── object-storage read-through (P1.8) ────────────────────────────────────

  /**
   * Enables object-storage read-through: subsequent {@link #scan} calls fetch this cluster's
   * vectors from {@code backend} (a network GET on the first probe — "cold") and cache the slice in
   * heap so later probes are served without a round-trip ("warm"). This makes per-query cold/warm
   * object-storage reads observable for benchmarking the object-storage-native serving path, rather
   * than always scanning the heap-resident global array. Idempotent; clears any cached slice.
   */
  public void enableReadThrough(StorageBackend backend) {
    // Publish backend + cleared cache in a single atomic update so concurrent readers cannot
    // observe a fresh backend with a stale cache from a previous enableReadThrough call.
    this.readThrough = new ReadThroughState(backend, null);
  }

  /**
   * Drops the cached read-through slice so the next {@link #scan} re-fetches (forces a cold read).
   */
  public void dropReadThroughCache() {
    ReadThroughState rt = this.readThrough;
    if (rt == null || rt.cache == null) return;
    this.readThrough = new ReadThroughState(rt.backend, null);
  }

  /** True when read-through is enabled and the cluster slice is cached in heap (warm). */
  public boolean isReadThroughWarm() {
    ReadThroughState rt = this.readThrough;
    return rt != null && rt.cache != null;
  }

  private List<IvfHit> scanReadThrough(float[] query, int k, float minScore, ReadThroughState rt) {
    float[][] slice = rt.cache;
    if (slice == null) {
      // Cold path — single-flight the network fetch so N concurrent probes don't trigger N
      // GETs against the object-storage backend (audit T1.3). The hot path above stays
      // lock-free.
      synchronized (readThroughLoadLock) {
        ReadThroughState current = this.readThrough;
        // Another thread may have populated the cache while we waited for the lock; re-read the
        // state and accept its slice if so. We also re-read the backend reference in case
        // enableReadThrough has been re-issued during the wait.
        if (current == null) {
          // Read-through was disabled while we waited — fall back to the heap-resident scan.
          return (t1Data != null) ? scanWithT1(query, k, minScore) : scanExact(query, k, minScore);
        }
        if (current.cache != null) {
          slice = current.cache;
        } else {
          try {
            slice = fetchT3(current.backend); // cold: network GET from object storage
          } catch (IOException e) {
            throw new UncheckedIOException("read-through fetchT3 failed for " + t3Key(), e);
          }
          this.readThrough = new ReadThroughState(current.backend, slice); // warm for next probe
        }
      }
    }
    return scanSlice(query, k, minScore, slice);
  }

  // ─── internals ─────────────────────────────────────────────────────────────

  private String t3Key() {
    return "gen-" + t3Generation + "/cluster-" + partition.clusterId();
  }

  private String t2Key() {
    return "gen-" + t3Generation + "/cluster-T2-" + partition.clusterId();
  }

  /**
   * Serialises this cluster's vectors to the given {@link StorageBackend} under the T2 key {@code
   * "cluster-T2-<clusterId>"}, representing a warm (local SSD / mmap) snapshot.
   */
  private void storeT2(StorageBackend backend) throws IOException {
    float[][] vecs = extractClusterVectors();
    backend.put(t2Key(), serializeVectors(vecs));
  }

  private float[][] extractClusterVectors() {
    int[] ordinals = partition.ordinals();
    float[][] vecs = new float[ordinals.length][];
    for (int i = 0; i < ordinals.length; i++) vecs[i] = globalVectors[ordinals[i]];
    return vecs;
  }

  /** Batch size for the fused GEMV scan; matched to {@link IvfIndex}'s partition scan. */
  private static final int BATCH_ROWS = 64;

  private List<IvfHit> scanExact(float[] query, int k, float minScore) {
    int[] ordinals = partition.ordinals();
    // extractClusterVectors()[i] == globalVectors[ordinals[i]], so ordinals[i] is the id to emit.
    return exactTopK(query, ordinals, extractClusterVectors(), ordinals.length, k, minScore);
  }

  private List<IvfHit> scanWithT1(float[] query, int k, float minScore) {
    int[] ordinals = partition.ordinals();
    var scoreFunction = t1Data.scoreFunctionFor(query, metric);
    // SQ8 scoring is delegated to the compressed-vector score function, so the batched
    // full-precision
    // GEMV kernels do not apply here; we still drop the per-candidate IvfHit boxing +
    // PriorityQueue.
    TopKHeap heap = new TopKHeap(k);
    for (int local = 0; local < ordinals.length; local++) {
      float s = scoreFunction.score(local);
      if (s < minScore) continue;
      heap.offer(ordinals[local], s);
    }
    return drainToHits(heap);
  }

  /** Exact scan over an externally-supplied cluster slice (read-through path). */
  private List<IvfHit> scanSlice(float[] query, int k, float minScore, float[][] slice) {
    int[] ordinals = partition.ordinals();
    int limit = Math.min(slice.length, ordinals.length);
    return exactTopK(query, ordinals, slice, limit, k, minScore);
  }

  /**
   * Fused top-{@code k} scan over the first {@code limit} full-precision rows of {@code slice},
   * where {@code slice[i]} is the vector for {@code ordinals[i]}. EUCLIDEAN uses {@link
   * VectorUtil#batchSquaredL2} and DOT/MIP uses {@link VectorUtil#batchDotProduct}; COSINE keeps
   * the exact scalar {@link VectorUtil#cosine} (bit-identical). A primitive {@link TopKHeap}
   * replaces the per-candidate {@code IvfHit} boxing + {@link java.util.PriorityQueue} of the
   * previous implementation.
   */
  private List<IvfHit> exactTopK(
      float[] query, int[] ordinals, float[][] slice, int limit, int k, float minScore) {
    TopKHeap heap = new TopKHeap(k);
    boolean euclidean = metric == SimilarityFunction.EUCLIDEAN;
    if (metric == SimilarityFunction.COSINE) {
      for (int local = 0; local < limit; local++) {
        float s = score(query, slice[local]);
        if (s < minScore) continue;
        heap.offer(ordinals[local], s);
      }
      return drainToHits(heap);
    }
    float[][] rows = new float[BATCH_ROWS][];
    float[] out = new float[BATCH_ROWS];
    for (int start = 0; start < limit; start += BATCH_ROWS) {
      int count = Math.min(BATCH_ROWS, limit - start);
      for (int j = 0; j < count; j++) rows[j] = slice[start + j];
      if (euclidean) {
        VectorUtil.batchSquaredL2(query, rows, out, count);
      } else {
        VectorUtil.batchDotProduct(query, rows, out, count);
      }
      for (int j = 0; j < count; j++) {
        float s = euclidean ? -out[j] : out[j];
        if (s < minScore) continue;
        heap.offer(ordinals[start + j], s);
      }
    }
    return drainToHits(heap);
  }

  private static List<IvfHit> drainToHits(TopKHeap heap) {
    TopKHeap.DrainResult drained = heap.drainDescending();
    int[] ords = drained.ordinals();
    float[] scores = drained.scores();
    List<IvfHit> hits = new ArrayList<>(ords.length);
    for (int i = 0; i < ords.length; i++) {
      hits.add(new IvfHit(ords[i], null, scores[i]));
    }
    return hits;
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
