package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * P23 gate test — HyperDoor cross-tier eviction and TieredClusterRegistry lookup.
 *
 * <p>Verifies that:
 *
 * <ul>
 *   <li>{@link TieredCluster#evictToTier(TierPolicy.Tier,
 *       com.integrallis.vectors.storage.backend.StorageBackend)} persists data to the correct
 *       backend key and clears the appropriate in-heap state.
 *   <li>{@link TieredClusterRegistry#lookup(HyperDoor)} finds a registered cluster and returns
 *       empty for an unknown door.
 * </ul>
 */
@Tag("unit")
class HyperDoorEvictionTest {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private ClusterPartition makePartition(int clusterId, int size) {
    int[] ordinals = new int[size];
    for (int i = 0; i < size; i++) ordinals[i] = i;
    return ClusterPartition.of(clusterId, new float[DIM], ordinals);
  }

  // ─── evictToTier ─────────────────────────────────────────────────────────

  /**
   * evictToTier(T2) must persist float32 data under the cluster's T2 key and clear the T1 in-heap
   * SQ8 data.
   */
  @Test
  void evictToT2_persistsToBackend_and_clearsT1() throws IOException {
    float[][] vecs = randomVecs(20, DIM, 1L);
    TieredCluster tc = new TieredCluster(makePartition(7, 20), vecs, METRIC);
    tc.materializeT1();
    assertThat(tc.hasT1()).isTrue();

    HeapStorageBackend backend = new HeapStorageBackend();
    tc.evictToTier(TierPolicy.Tier.T2, backend);

    assertThat(backend.get("cluster-T2-7")).isNotNull();
    assertThat(tc.hasT1()).isFalse();
  }

  /**
   * evictToTier(T3) must persist float32 data under the cluster's T3 key and clear both T1 in-heap
   * data and any in-flight T2 state.
   */
  @Test
  void evictToT3_persistsToT3_and_clearsT1andT2() throws IOException {
    float[][] vecs = randomVecs(20, DIM, 2L);
    TieredCluster tc = new TieredCluster(makePartition(3, 20), vecs, METRIC);
    tc.materializeT1();
    assertThat(tc.hasT1()).isTrue();

    HeapStorageBackend backend = new HeapStorageBackend();
    tc.evictToTier(TierPolicy.Tier.T3, backend);

    assertThat(backend.get("cluster-3")).isNotNull();
    assertThat(tc.hasT1()).isFalse();
  }

  // ─── TieredClusterRegistry ───────────────────────────────────────────────

  /** Registry must return the registered cluster when looked up by its hyperDoor. */
  @Test
  void lookup_returnsClusterForKnownHyperDoor() {
    float[][] vecs = randomVecs(10, DIM, 3L);
    TieredCluster tc = new TieredCluster(makePartition(5, 10), vecs, METRIC);

    TieredClusterRegistry registry = new TieredClusterRegistry();
    registry.register(tc);

    HyperDoor door = tc.hyperDoor();
    Optional<TieredCluster> found = registry.lookup(door);

    assertThat(found).isPresent().contains(tc);
  }

  /** Registry must return empty for a HyperDoor with an unregistered cluster ordinal. */
  @Test
  void lookup_returnsEmptyForUnknownHyperDoor() {
    TieredClusterRegistry registry = new TieredClusterRegistry();
    HyperDoor unknown = HyperDoor.t0AndT3Only(99, DIM);
    assertThat(registry.lookup(unknown)).isEmpty();
  }
}
