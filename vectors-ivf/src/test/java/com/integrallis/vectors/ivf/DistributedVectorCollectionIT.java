package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration / durability gate tests for {@link DistributedVectorCollection}.
 *
 * <p>These tests exercise the WAL-replay path, cold T3 cluster restoration, and conditional-put
 * double-commit prevention. They use in-process {@link HeapStorageBackend} to simulate T3 (S3); a
 * real S3-backed variant using TestContainers/LocalStack will be added when a production {@code
 * S3StorageBackend} is implemented.
 *
 * <p>Tagged {@code @Tag("integration")} — excluded from the default build; run via {@code ./gradlew
 * integrationTest}.
 */
@Tag("integration")
class DistributedVectorCollectionIT {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private String[] ids(int n) {
    String[] a = new String[n];
    for (int i = 0; i < n; i++) a[i] = "doc-" + i;
    return a;
  }

  private DistributedVectorCollection build(
      float[][] vecs, String[] docIds, Path walDir, StorageBackend t3) throws IOException {
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, 42L);
    TierPolicy policy = new TierPolicy(5, 2);
    return DistributedVectorCollection.build(
        vecs, docIds, METRIC, params, splitter, policy, walDir, t3);
  }

  // ─── WAL replay ──────────────────────────────────────────────────────────

  /**
   * Simulates a crash-recovery cycle: build → add → commit → close → reopen. After replay the
   * collection must find committed vectors.
   */
  @Test
  void walReplay_restoresCommittedVectors(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] initial = randomVecs(100, DIM, 1L);

    // Session 1: build, add, commit, close
    try (var col = build(initial, ids(100), walDir, t3)) {
      float[] needle = new float[DIM];
      Arrays.fill(needle, 80f);
      col.add("needle", needle);
      col.commit();
    } // WAL is closed / sealed here

    // Session 2: reopen from WAL and verify needle is found
    try (var col = DistributedVectorCollection.open(walDir, METRIC, new TierPolicy(5, 2), t3)) {
      float[] needle = new float[DIM];
      Arrays.fill(needle, 80f);
      List<IvfHit> hits = col.search(needle, 1, 4);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }

  @Test
  void walReplay_restoresGenerationCounter(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] vecs = randomVecs(100, DIM, 2L);

    long genAfterTwoCommits;
    try (var col = build(vecs, ids(100), walDir, t3)) {
      col.add("v1", randomVecs(1, DIM, 10L)[0]);
      col.commit();
      col.add("v2", randomVecs(1, DIM, 11L)[0]);
      col.commit();
      genAfterTwoCommits = col.generation();
    }

    try (var col = DistributedVectorCollection.open(walDir, METRIC, new TierPolicy(5, 2), t3)) {
      assertThat(col.generation()).isEqualTo(genAfterTwoCommits);
    }
  }

  // ─── T3 cold fetch ───────────────────────────────────────────────────────

  @Test
  void t3Backend_storesClusterSnapshots(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] vecs = randomVecs(80, DIM, 3L);
    try (var col = build(vecs, ids(80), walDir, t3)) {
      col.commit();
    }
    // Each cluster should have a snapshot stored in T3
    List<String> keys = t3.list("cluster-");
    assertThat(keys).isNotEmpty();
  }

  @Test
  void t3Backend_routingIndexIsPersisted(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] vecs = randomVecs(80, DIM, 4L);
    try (var col = build(vecs, ids(80), walDir, t3)) {
      // routing index must be persisted immediately on build; reference col to satisfy compiler
      assertThat(col.size()).isGreaterThan(0);
    }
    assertThat(t3.get("_routing-index")).isNotNull();
  }

  // ─── conditional-put double-commit prevention ────────────────────────────

  @Test
  void conditionalPut_preventsDoubleCommit(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] vecs = randomVecs(50, DIM, 5L);
    try (var col = build(vecs, ids(50), walDir, t3)) {
      col.commit(); // writes cluster snapshots via storeT3
    }

    // Verify the cluster key is already present (written by build → commit → storeT3)
    String key = "cluster-0";
    byte[] firstBytes = t3.get(key);
    assertThat(firstBytes).isNotNull();

    // A conditional PUT with expectedEtag=null means "must not exist" — must be rejected
    // because the key already exists.
    StorageBackend.ConditionalPutResult r = t3.conditionalPut(key, new byte[] {1, 2, 3}, null);
    assertThat(r.succeeded()).isFalse();

    // The stored value must be unchanged
    assertThat(t3.get(key)).isEqualTo(firstBytes);
  }

  // ─── size after replay ───────────────────────────────────────────────────

  @Test
  void walReplay_sizeMatchesCommittedVectors(@TempDir Path walDir) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    float[][] vecs = randomVecs(100, DIM, 6L);
    int expectedSize;

    try (var col = build(vecs, ids(100), walDir, t3)) {
      col.add("extra1", randomVecs(1, DIM, 20L)[0]);
      col.add("extra2", randomVecs(1, DIM, 21L)[0]);
      col.commit();
      expectedSize = col.size();
    }

    try (var col = DistributedVectorCollection.open(walDir, METRIC, new TierPolicy(5, 2), t3)) {
      assertThat(col.size()).isEqualTo(expectedSize);
    }
  }
}
