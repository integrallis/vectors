package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** P14: TieredCluster — T1 SQ8 materialization, T3 float32 round-trip, access tracking. */
@Tag("unit")
class TieredClusterTest {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  /** Build a ClusterPartition for the first `size` vectors in allVectors. */
  private ClusterPartition makePartition(int clusterId, int size) {
    int[] ordinals = new int[size];
    for (int i = 0; i < size; i++) ordinals[i] = i;
    return new ClusterPartition(clusterId, new float[DIM], ordinals, size);
  }

  // ─── access tracking ─────────────────────────────────────────────────────

  @Test
  void newCluster_hasZeroAccessCount() {
    float[][] vecs = randomVecs(20, DIM, 1L);
    TieredCluster tc = new TieredCluster(makePartition(0, 20), vecs, METRIC);
    assertThat(tc.accessCount()).isZero();
  }

  @Test
  void recordAccess_incrementsCount() {
    float[][] vecs = randomVecs(20, DIM, 2L);
    TieredCluster tc = new TieredCluster(makePartition(0, 20), vecs, METRIC);
    tc.recordAccess();
    tc.recordAccess();
    tc.recordAccess();
    assertThat(tc.accessCount()).isEqualTo(3);
  }

  // ─── T1 materialization ──────────────────────────────────────────────────

  @Test
  void newCluster_t1IsAbsent() {
    float[][] vecs = randomVecs(20, DIM, 3L);
    TieredCluster tc = new TieredCluster(makePartition(0, 20), vecs, METRIC);
    assertThat(tc.hasT1()).isFalse();
  }

  @Test
  void materializeT1_setsHasT1() {
    float[][] vecs = randomVecs(30, DIM, 4L);
    TieredCluster tc = new TieredCluster(makePartition(0, 30), vecs, METRIC);
    tc.materializeT1();
    assertThat(tc.hasT1()).isTrue();
  }

  @Test
  void evictT1_clearsHasT1() {
    float[][] vecs = randomVecs(30, DIM, 5L);
    TieredCluster tc = new TieredCluster(makePartition(0, 30), vecs, METRIC);
    tc.materializeT1();
    assertThat(tc.hasT1()).isTrue();
    tc.evictT1();
    assertThat(tc.hasT1()).isFalse();
  }

  // ─── T3 round-trip ────────────────────────────────────────────────────────

  @Test
  void storeT3AndFetch_roundTrips() throws IOException {
    float[][] vecs = randomVecs(20, DIM, 6L);
    ClusterPartition partition = makePartition(0, 20);
    TieredCluster tc = new TieredCluster(partition, vecs, METRIC);

    HeapStorageBackend backend = new HeapStorageBackend();
    tc.storeT3(backend);
    float[][] fetched = tc.fetchT3(backend);

    assertThat(fetched.length).isEqualTo(20);
    for (int i = 0; i < 20; i++) {
      for (int d = 0; d < DIM; d++) {
        assertThat(fetched[i][d]).isCloseTo(vecs[i][d], org.assertj.core.data.Offset.offset(1e-5f));
      }
    }
  }

  @Test
  void storeT3_usesClusterIdAsKey() throws IOException {
    float[][] vecs = randomVecs(10, DIM, 7L);
    TieredCluster tc = new TieredCluster(makePartition(42, 10), vecs, METRIC);
    HeapStorageBackend backend = new HeapStorageBackend();
    tc.storeT3(backend);
    assertThat(backend.get("cluster-42")).isNotNull();
  }

  // ─── scan ─────────────────────────────────────────────────────────────────

  @Test
  void scan_returnsAtMostKHits() {
    float[][] vecs = randomVecs(50, DIM, 8L);
    TieredCluster tc = new TieredCluster(makePartition(0, 50), vecs, METRIC);
    float[] query = randomVecs(1, DIM, 99L)[0];

    List<IvfHit> hits = tc.scan(query, 10, -Float.MAX_VALUE);
    assertThat(hits.size()).isLessThanOrEqualTo(10);
  }

  @Test
  void scan_hitsAreInDescendingScoreOrder() {
    float[][] vecs = randomVecs(50, DIM, 9L);
    TieredCluster tc = new TieredCluster(makePartition(0, 50), vecs, METRIC);
    float[] query = randomVecs(1, DIM, 100L)[0];

    List<IvfHit> hits = tc.scan(query, 10, -Float.MAX_VALUE);
    for (int i = 0; i < hits.size() - 1; i++) {
      assertThat(hits.get(i).score()).isGreaterThanOrEqualTo(hits.get(i + 1).score());
    }
  }

  @Test
  void scanWithT1_doesNotThrowAndReturnsResults() {
    float[][] vecs = randomVecs(50, DIM, 10L);
    TieredCluster tc = new TieredCluster(makePartition(0, 50), vecs, METRIC);
    tc.materializeT1();
    float[] query = randomVecs(1, DIM, 101L)[0];

    List<IvfHit> hits = tc.scan(query, 10, -Float.MAX_VALUE);
    assertThat(hits).isNotEmpty();
    assertThat(hits.size()).isLessThanOrEqualTo(10);
  }

  @Test
  void scanWithT1_returnsGlobalOrdinals() {
    float[][] vecs = randomVecs(30, DIM, 11L);
    // Partition references ordinals 0..29
    TieredCluster tc = new TieredCluster(makePartition(0, 30), vecs, METRIC);
    tc.materializeT1();
    float[] query = randomVecs(1, DIM, 102L)[0];

    List<IvfHit> hits = tc.scan(query, 5, -Float.MAX_VALUE);
    for (IvfHit hit : hits) {
      assertThat(hit.ordinal()).isBetween(0, 29);
    }
  }

  // ─── C1: bounded allocation ─────────────────────────────────────────────

  @Test
  void scanExact_largeCluster_returnsExactlyK() {
    // 5000-vector cluster requesting k=5 — must not allocate a 5000-element list internally
    int n = 5000;
    float[][] vecs = randomVecs(n, DIM, 50L);
    TieredCluster tc = new TieredCluster(makePartition(0, n), vecs, METRIC);
    float[] query = randomVecs(1, DIM, 51L)[0];

    List<IvfHit> hits = tc.scan(query, 5, -Float.MAX_VALUE);
    assertThat(hits).hasSize(5);
    // Verify descending order
    for (int i = 0; i < hits.size() - 1; i++) {
      assertThat(hits.get(i).score()).isGreaterThanOrEqualTo(hits.get(i + 1).score());
    }
  }

  @Test
  void scanExact_returnsTopKByScore_notArbitraryK() {
    // Place a known "needle" vector close to the query; verify it always appears in top-k
    float[][] vecs = randomVecs(200, DIM, 60L);
    float[] query = randomVecs(1, DIM, 61L)[0];
    // Make ordinal 42 a copy of the query (distance = 0, best possible hit)
    System.arraycopy(query, 0, vecs[42], 0, DIM);

    TieredCluster tc = new TieredCluster(makePartition(0, 200), vecs, METRIC);
    List<IvfHit> hits = tc.scan(query, 5, -Float.MAX_VALUE);

    assertThat(hits.get(0).ordinal()).isEqualTo(42);
    assertThat(hits.get(0).score()).isCloseTo(0f, org.assertj.core.data.Offset.offset(1e-4f));
  }

  @Test
  void scanWithT1_largeCluster_returnsExactlyK() {
    int n = 5000;
    float[][] vecs = randomVecs(n, DIM, 70L);
    TieredCluster tc = new TieredCluster(makePartition(0, n), vecs, METRIC);
    tc.materializeT1();
    float[] query = randomVecs(1, DIM, 71L)[0];

    List<IvfHit> hits = tc.scan(query, 5, -Float.MAX_VALUE);
    assertThat(hits).hasSize(5);
    for (int i = 0; i < hits.size() - 1; i++) {
      assertThat(hits.get(i).score()).isGreaterThanOrEqualTo(hits.get(i + 1).score());
    }
  }
}
