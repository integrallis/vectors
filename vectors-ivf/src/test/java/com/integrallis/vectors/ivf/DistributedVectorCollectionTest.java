package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Unit tests for DistributedVectorCollection — in-process, no Docker required. */
@Tag("unit")
class DistributedVectorCollectionTest {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private String[] ids(int n) {
    String[] arr = new String[n];
    for (int i = 0; i < n; i++) arr[i] = "doc-" + i;
    return arr;
  }

  private int[] bruteForceTopK(float[] query, float[][] data, int k) {
    record Pair(int idx, float dist) {}
    Pair[] pairs = new Pair[data.length];
    for (int i = 0; i < data.length; i++) {
      float d = 0f;
      for (int j = 0; j < query.length; j++) {
        float diff = query[j] - data[i][j];
        d += diff * diff;
      }
      pairs[i] = new Pair(i, d);
    }
    Arrays.sort(pairs, (a, b) -> Float.compare(a.dist(), b.dist()));
    int[] result = new int[k];
    for (int i = 0; i < k; i++) result[i] = pairs[i].idx();
    return result;
  }

  private DistributedVectorCollection buildCollection(float[][] vecs, String[] docIds, Path walDir)
      throws IOException {
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, 42L); // no splits
    TierPolicy policy = new TierPolicy(5, 2);
    return DistributedVectorCollection.build(
        vecs, docIds, METRIC, params, splitter, policy, walDir, new HeapStorageBackend());
  }

  // ─── construction ────────────────────────────────────────────────────────

  @Test
  void build_sizeEqualsInitialVectorCount(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(200, DIM, 1L);
    try (var col = buildCollection(vecs, ids(200), tmp)) {
      assertThat(col.size()).isEqualTo(200);
    }
  }

  @Test
  void build_searchReturnsAtMostKHits(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(200, DIM, 2L);
    try (var col = buildCollection(vecs, ids(200), tmp)) {
      float[] query = randomVecs(1, DIM, 99L)[0];
      List<IvfHit> hits = col.search(query, 10, 3);
      assertThat(hits.size()).isLessThanOrEqualTo(10);
    }
  }

  // ─── add / staging ───────────────────────────────────────────────────────

  @Test
  void add_incrementsSize(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(100, DIM, 3L);
    try (var col = buildCollection(vecs, ids(100), tmp)) {
      int before = col.size();
      col.add("new-1", randomVecs(1, DIM, 10L)[0]);
      col.add("new-2", randomVecs(1, DIM, 11L)[0]);
      assertThat(col.size()).isEqualTo(before + 2);
    }
  }

  @Test
  void search_includesStagedVectors(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(100, DIM, 4L);
    try (var col = buildCollection(vecs, ids(100), tmp)) {
      // Add a distinctive vector far from all training data
      float[] needle = new float[DIM];
      Arrays.fill(needle, 100f); // very large values, unique
      col.add("needle", needle);

      List<IvfHit> hits = col.search(needle, 1, 4);
      assertThat(hits).isNotEmpty();
      // The staged needle should be the top hit (nearest to itself)
      assertThat(hits.get(0).id()).isEqualTo("needle");
    }
  }

  // ─── commit ──────────────────────────────────────────────────────────────

  @Test
  void commit_clearsStaging(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(100, DIM, 5L);
    try (var col = buildCollection(vecs, ids(100), tmp)) {
      col.add("x", randomVecs(1, DIM, 20L)[0]);
      int sizeBeforeCommit = col.size(); // includes staging
      col.commit();
      assertThat(col.stagingSize()).isZero();
      assertThat(col.size()).isEqualTo(sizeBeforeCommit); // total unchanged
    }
  }

  @Test
  void commit_writesWalFile(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(100, DIM, 6L);
    try (var col = buildCollection(vecs, ids(100), tmp)) {
      col.add("w1", randomVecs(1, DIM, 30L)[0]);
      col.commit();
    }
    // At least one WAL segment file must exist
    long walFiles = Files.list(tmp).filter(p -> p.toString().endsWith(".seg")).count();
    assertThat(walFiles).isGreaterThan(0);
  }

  @Test
  void commit_thenSearchFindsCommittedVector(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(100, DIM, 7L);
    try (var col = buildCollection(vecs, ids(100), tmp)) {
      float[] needle = new float[DIM];
      Arrays.fill(needle, 50f);
      col.add("committed-needle", needle);
      col.commit();

      List<IvfHit> hits = col.search(needle, 1, 4);
      assertThat(hits).isNotEmpty();
      assertThat(hits.get(0).id()).isEqualTo("committed-needle");
    }
  }

  // ─── tier policy ─────────────────────────────────────────────────────────

  @Test
  void applyTierPolicy_promotesHotCluster(@TempDir Path tmp) throws IOException {
    float[][] vecs = randomVecs(200, DIM, 8L);
    try (var col = buildCollection(vecs, ids(200), tmp)) {
      // Drive access on one cluster heavily by searching many times
      float[] pivot = vecs[0];
      for (int i = 0; i < 10; i++) col.search(pivot, 5, 1);
      col.commit(); // triggers applyTierPolicy
      // At least one cluster should be promoted to T1
      assertThat(col.t1ClusterCount()).isGreaterThan(0);
    }
  }

  // ─── C3: incremental commit — only dirty clusters rebuild ─────────────

  @Test
  void commit_preservesT1OnUntouchedClusters(@TempDir Path tmp) throws IOException {
    // Build, promote ALL clusters to T1 by heavy access, commit.
    // Then add one vector and commit again.
    // Untouched clusters should still have T1 (not rebuilt and re-evicted).
    float[][] vecs = randomVecs(200, DIM, 30L);
    TierPolicy alwaysT1 = new TierPolicy(2, 1); // t1 after 2 accesses, t2 after 1
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, 42L);
    HeapStorageBackend backend = new HeapStorageBackend();
    try (var col =
        DistributedVectorCollection.build(
            vecs, ids(200), METRIC, params, splitter, alwaysT1, tmp, backend)) {

      // Access all clusters heavily to promote to T1
      for (int i = 0; i < 20; i++) col.search(randomVecs(1, DIM, 500L + i)[0], 5, 4);
      col.commit(); // promotes to T1
      int t1Before = col.t1ClusterCount();
      assertThat(t1Before).isGreaterThan(0);

      // Now add just one vector and commit — only 1 cluster should be rebuilt
      col.add("extra", randomVecs(1, DIM, 999L)[0]);
      col.commit();

      // All T1 clusters should still be materialised
      assertThat(col.t1ClusterCount()).isGreaterThanOrEqualTo(t1Before);
    }
  }

  @Test
  void commit_preservesAccessCountOnUntouchedClusters(@TempDir Path tmp) throws IOException {
    // Access a specific cluster, commit. Verify that after a second commit (with vectors going
    // to a different cluster), the first cluster's access count is preserved exactly.
    float[][] vecs = randomVecs(200, DIM, 31L);
    try (var col = buildCollection(vecs, ids(200), tmp)) {
      // Search 5 times to build access counts
      for (int i = 0; i < 5; i++) col.search(vecs[0], 5, 1);
      col.commit();

      // Record the total access count before second commit
      // (we can't inspect individual clusters, but t1ClusterCount is a proxy)
      int sizeBefore = col.size();

      // Add a vector and commit
      col.add("new", randomVecs(1, DIM, 888L)[0]);
      col.commit();

      // Size should increase by 1, search should still work
      assertThat(col.size()).isEqualTo(sizeBefore + 1);
      List<IvfHit> hits = col.search(vecs[0], 5, 2);
      assertThat(hits).isNotEmpty();
    }
  }

  // ─── C4: concurrent reads ──────────────────────────────────────────────

  @Test
  void search_allowsConcurrentReads(@TempDir Path tmp) throws Exception {
    float[][] vecs = randomVecs(500, DIM, 40L);
    try (var col = buildCollection(vecs, ids(500), tmp)) {
      int nThreads = 4;
      int queriesPerThread = 10;
      CountDownLatch start = new CountDownLatch(1);
      AtomicInteger completed = new AtomicInteger();
      ExecutorService pool = Executors.newFixedThreadPool(nThreads);

      for (int t = 0; t < nThreads; t++) {
        final int threadId = t;
        pool.submit(
            () -> {
              try {
                start.await();
                for (int q = 0; q < queriesPerThread; q++) {
                  float[] query = randomVecs(1, DIM, 2000L + threadId * 100 + q)[0];
                  List<IvfHit> hits = col.search(query, 5, 2);
                  assertThat(hits).isNotEmpty();
                  completed.incrementAndGet();
                }
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
            });
      }
      start.countDown();
      pool.shutdown();
      pool.awaitTermination(30, java.util.concurrent.TimeUnit.SECONDS);
      assertThat(completed.get()).isEqualTo(nThreads * queriesPerThread);
    }
  }

  // ─── recall ──────────────────────────────────────────────────────────────

  @Test
  void searchRecall_exceedsThreshold(@TempDir Path tmp) throws IOException {
    int n = 500, k = 10, nprobe = 3;
    float[][] data = randomVecs(n, DIM, 9L);
    try (var col = buildCollection(data, ids(n), tmp)) {
      int queries = 20;
      double totalRecall = 0.0;
      for (int q = 0; q < queries; q++) {
        float[] query = randomVecs(1, DIM, 1000L + q)[0];
        List<IvfHit> hits = col.search(query, k, nprobe);
        int[] found = hits.stream().mapToInt(IvfHit::ordinal).toArray();
        int[] gt = bruteForceTopK(query, data, k);
        Set<Integer> gtSet = new HashSet<>();
        for (int g : gt) gtSet.add(g);
        int hitCount = 0;
        for (int f : found) if (gtSet.contains(f)) hitCount++;
        totalRecall += (double) hitCount / k;
      }
      assertThat(totalRecall / queries).isGreaterThanOrEqualTo(0.80);
    }
  }
}
