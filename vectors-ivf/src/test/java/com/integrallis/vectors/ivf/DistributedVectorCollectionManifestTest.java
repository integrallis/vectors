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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.manifest.ManifestStore;
import com.integrallis.vectors.storage.manifest.StorageManifest;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies DartVault's durable object-storage generation manifest: {@code build}/{@code commit}
 * publish a CAS'd, monotonic {@code _manifest} pointer so a remote reader can discover the live
 * generation ({@code open} resolves from it), and commits announce the generation to a gossip
 * bridge. The WAL stays the local recovery authority; the manifest is the discoverable layer.
 */
@Tag("unit")
class DistributedVectorCollectionManifestTest {

  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;
  private static final String MANIFEST_KEY = "_manifest";
  private static final int DIM = 16;

  private static float[][] randomVectors(int n, long seed) {
    Random rng = new Random(seed);
    float[][] v = new float[n][DIM];
    for (float[] row : v) for (int d = 0; d < DIM; d++) row[d] = rng.nextFloat();
    return v;
  }

  private static DistributedVectorCollection build(HeapStorageBackend t3, Path walDir, int n)
      throws IOException {
    float[][] vecs = randomVectors(n, 1L);
    String[] ids = new String[n];
    for (int i = 0; i < n; i++) ids[i] = "doc-" + i;
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(10_000, 30, 42L);
    TierPolicy policy = new TierPolicy(5, 2);
    return DistributedVectorCollection.build(
        vecs, ids, METRIC, params, splitter, policy, walDir, t3);
  }

  private static StorageManifest manifest(HeapStorageBackend t3) throws IOException {
    return new ManifestStore(t3, MANIFEST_KEY).load().manifest();
  }

  @Test
  void buildPublishesGenerationZeroManifest(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      assertThat(col.generation()).isEqualTo(0L);
      StorageManifest m = manifest(t3);
      assertThat(m.isEmpty()).isFalse();
      assertThat(m.generation()).isEqualTo(0L);
      // Per-cluster entries: k=4 clusters (min(params.k()=4, n=20)), all at generation 0.
      assertThat(m.entries()).containsOnlyKeys("cluster-0", "cluster-1", "cluster-2", "cluster-3");
      assertThat(m.entries().values()).allMatch(g -> g == 0L);
    }
  }

  @Test
  void commitAdvancesOnlyTheClustersItRewrites(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      // Add a single vector: only its owning cluster is dirty and advances to generation 1; the
      // rest
      // stay at generation 0 (no write amplification, and their gen-0 objects remain intact).
      col.add("solo", randomVectors(1, 77L)[0]);
      col.commit();

      StorageManifest m = manifest(t3);
      assertThat(m.generation()).isEqualTo(1L);
      long advanced = m.entries().values().stream().filter(g -> g == 1L).count();
      long unchanged = m.entries().values().stream().filter(g -> g == 0L).count();
      assertThat(advanced).as("exactly the dirty clusters advanced").isBetween(1L, 4L);
      assertThat(advanced + unchanged).isEqualTo(4L);
      assertThat(unchanged)
          .as("untouched clusters kept their generation-0 objects")
          .isGreaterThan(0L);

      // The prior generation's payload objects are still present (crash-atomic / time-travel):
      // a partial write of the new generation could never have clobbered them.
      assertThat(t3.get("gen-0/cluster-0")).isNotNull();
    }
  }

  @Test
  void commitAdvancesTheDurableManifestGeneration(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      col.add("new-1", randomVectors(1, 9L)[0]);
      col.commit();
      assertThat(col.generation()).isEqualTo(1L);
      assertThat(manifest(t3).generation()).isEqualTo(1L);

      col.add("new-2", randomVectors(1, 10L)[0]);
      col.commit();
      assertThat(col.generation()).isEqualTo(2L);
      assertThat(manifest(t3).generation()).isEqualTo(2L);
    }
  }

  @Test
  void commitsAnnounceMonotonicGenerations(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    List<Long> announced = new ArrayList<>();
    try (var col = build(t3, tmp, 20)) {
      col.setGenerationAnnouncer((gen, hash) -> announced.add(gen));
      col.add("a", randomVectors(1, 2L)[0]);
      col.commit();
      col.add("b", randomVectors(1, 3L)[0]);
      col.commit();
    }
    // The gossip bridge sees strictly-increasing generations — exactly what announceVersion's
    // max-wins guard accepts; a follower fed this never reloads an older index.
    assertThat(announced).containsExactly(1L, 2L);
  }

  @Test
  void writerIdIsRecordedAsManifestProvenance(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      col.setWriterId("node-7");
      col.add("a", randomVectors(1, 5L)[0]);
      col.commit();
    }
    assertThat(manifest(t3).writer()).isEqualTo("node-7");
  }

  @Test
  void openResolvesGenerationFromTheManifest(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      col.add("x", randomVectors(1, 4L)[0]);
      col.commit(); // generation 1, published to WAL + manifest
    }
    // Reopen against the same WAL + object storage: the committed generation is recovered.
    TierPolicy policy = new TierPolicy(5, 2);
    try (var reopened = DistributedVectorCollection.open(tmp, METRIC, policy, t3)) {
      assertThat(reopened.generation()).isEqualTo(1L);
    }
  }

  @Test
  void openDiscoversAManifestAheadOfTheLocalWal(@TempDir Path tmp) throws IOException {
    HeapStorageBackend t3 = new HeapStorageBackend();
    try (var col = build(t3, tmp, 20)) {
      assertThat(col.generation()).isEqualTo(0L); // WAL + manifest at 0
    }
    // Simulate a remote writer having published a newer generation to the shared manifest while
    // this
    // reader's local WAL is still at 0. open() must discover it (remote generation resolution).
    ManifestStore store = new ManifestStore(t3, MANIFEST_KEY);
    var loaded = store.load();
    store.compareAndPut(
        new StorageManifest(5L, "h5", Map.of("collection", 5L), 0L, "remote-writer"),
        loaded.etag());

    TierPolicy policy = new TierPolicy(5, 2);
    try (var reopened = DistributedVectorCollection.open(tmp, METRIC, policy, t3)) {
      assertThat(reopened.generation()).as("open resolved the manifest generation").isEqualTo(5L);
    }
  }
}
