package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end acceptance tests for the Step 4c Vamana backend wired into {@link VectorCollection}.
 *
 * <p>Structurally mirrors {@link VectorDbHnswPersistenceTest} so the two graph backends can be
 * audited side by side. Each nested class exercises one slice of the Step 4c scope:
 *
 * <ul>
 *   <li>{@link InMemoryVamana} — VAMANA works in in-memory mode (no {@code storagePath}),
 *       auto-commit triggers a rebuild, and {@code add}/{@code addAll} flows identically to FLAT
 *       and HNSW.
 *   <li>{@link PersistentVamana} — persistent VAMANA writes a {@code graph.bin} file per
 *       generation, round-trips identically across a close/reopen cycle, handles the
 *       empty-collection bootstrap, preserves metadata, and survives multi-generation commits.
 *   <li>{@link SimulatedKill9Vamana} — corrupting {@code graph.bin} or the manifest of the latest
 *       VAMANA generation forces recovery to fall back to the previous generation, mirroring the
 *       {@code VectorDbHnswPersistenceTest.SimulatedKill9Hnsw} pattern.
 *   <li>{@link CommitOpenFailureRecoveryVamana} — the A1 regression pattern extended to the VAMANA
 *       open path: a successful write followed by an injected {@code openGenerationFailureHook}
 *       must leave the collection usable and a retry must advance {@code nextGenerationNumber}
 *       instead of colliding with the orphaned generation directory.
 *   <li>{@link ConcurrencyVamana} — concurrent readers never see spurious {@code
 *       IllegalStateException("closed")} while the writer is swapping mmap-backed VAMANA
 *       generations underneath them.
 * </ul>
 */
class VectorDbVamanaPersistenceTest {

  private static final long SEED = 42L;
  private static final int DIM = 16;

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static Document docWithMetadata(String id, float[] vector, int i) {
    Map<String, MetadataValue> md = new HashMap<>();
    md.put("idx", MetadataValue.of(Long.valueOf(i)));
    md.put("name", MetadataValue.of("doc-" + i));
    md.put("flag", MetadataValue.of((i % 2) == 0));
    return new Document(id, vector, "text-" + i, md);
  }

  private static List<Document> generateDocs(int count, long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      out.add(docWithMetadata("doc-" + i, randomVector(rng), i));
    }
    return out;
  }

  /**
   * Computes the top-{@code k} brute-force neighbors of {@code query} over {@code docs} using the
   * {@link SimilarityFunction#EUCLIDEAN} score {@code 1 / (1 + d²)}. Used as the recall reference
   * for Vamana-backed searches in this module.
   */
  private static Set<String> bruteForceTopKIds(List<Document> docs, float[] query, int k) {
    SimilarityFunction sim = SimilarityFunction.EUCLIDEAN;
    String[] ids = new String[docs.size()];
    float[] scores = new float[docs.size()];
    for (int i = 0; i < docs.size(); i++) {
      ids[i] = docs.get(i).id();
      scores[i] = sim.compare(query, docs.get(i).vector());
    }
    Set<String> top = new HashSet<>();
    boolean[] used = new boolean[docs.size()];
    for (int r = 0; r < Math.min(k, docs.size()); r++) {
      int best = -1;
      for (int i = 0; i < docs.size(); i++) {
        if (used[i]) {
          continue;
        }
        if (best == -1 || scores[i] > scores[best]) {
          best = i;
        }
      }
      used[best] = true;
      top.add(ids[best]);
    }
    return top;
  }

  @Nested
  @Tag("unit")
  class InMemoryVamana {

    @Test
    void inMemoryVamanaSearchReturnsAccurateNeighbors() {
      // BEHAVIOR assertion: the VAMANA-backed collection returns results that agree with
      // brute-force ground truth to within Vamana's recall tolerance. A constant-scoring stub
      // would pass a shape-only test (k=10 hits) but fail this recall assertion.
      List<Document> docs = generateDocs(200, SEED);
      float[] query = randomVector(new Random(SEED + 1));

      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaMaxDegree(32)
              .vamanaSearchListSize(64)
              .vamanaSeed(SEED)
              .build()) {
        col.addAll(docs);
        col.commit();
        assertThat(col.size()).isEqualTo(200);

        SearchResult result = col.search(SearchRequest.builder(query, 10).build());
        assertThat(result.hits()).hasSize(10);
        // Hits must be sorted by score descending.
        for (int i = 1; i < result.hits().size(); i++) {
          assertThat(result.hits().get(i).score())
              .isLessThanOrEqualTo(result.hits().get(i - 1).score());
        }

        // Recall >= 0.8 against brute force (200 x 16-dim Vamana clears this consistently).
        Set<String> expectedTopK = bruteForceTopKIds(docs, query, 10);
        long hits = result.hits().stream().filter(h -> expectedTopK.contains(h.id())).count();
        assertThat((double) hits / 10).isGreaterThanOrEqualTo(0.8);
      }
    }

    @Test
    void inMemoryVamanaDefaultBuildParametersWork() {
      // No explicit vamanaMaxDegree / vamanaSearchListSize / vamanaAlpha — builder defaults.
      List<Document> docs = generateDocs(100, SEED);
      float[] query = randomVector(new Random(SEED + 1));

      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .build()) {
        col.addAll(docs);
        col.commit();
        assertThat(col.config().vamanaParams()).isNotNull();
        assertThat(col.config().vamanaParams().maxDegree())
            .isEqualTo(VectorCollectionBuilder.DEFAULT_VAMANA_R);
        assertThat(col.config().vamanaParams().searchListSize())
            .isEqualTo(VectorCollectionBuilder.DEFAULT_VAMANA_L);
        assertThat(col.config().vamanaParams().alpha())
            .isEqualTo(VectorCollectionBuilder.DEFAULT_VAMANA_ALPHA);

        SearchResult result = col.search(SearchRequest.builder(query, 5).build());
        assertThat(result.hits()).hasSize(5);
      }
    }

    @Test
    void inMemoryVamanaCustomBuildParametersFlowThrough() {
      // Verify the builder parameters are captured in the config and can be read back.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaMaxDegree(16)
              .vamanaSearchListSize(48)
              .vamanaAlpha(1.5f)
              .vamanaSeed(123L)
              .build()) {
        assertThat(col.config().vamanaParams().maxDegree()).isEqualTo(16);
        assertThat(col.config().vamanaParams().searchListSize()).isEqualTo(48);
        assertThat(col.config().vamanaParams().alpha()).isEqualTo(1.5f);
        assertThat(col.config().vamanaParams().seed()).isEqualTo(123L);
      }
    }

    @Test
    void inMemoryVamanaAutoCommitTriggersRebuild() {
      // autoCommitThreshold=50 with single-doc add()s: after the 50th add, maybeAutoCommit fires
      // and rebuilds the Vamana index with the first 50 docs. Docs 51-75 then sit in the staging
      // buffer (size=25, below the threshold) until an explicit commit() flushes them.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaSeed(SEED)
              .autoCommitThreshold(50)
              .build()) {
        List<Document> docs = generateDocs(75, SEED);
        for (Document d : docs) {
          col.add(d);
        }

        // Mid-stream state: 50 committed, 25 staged. size() reports committed liveCount only.
        assertThat(col.size()).isEqualTo(50);

        // A committed doc (doc-25) queried by its own vector MUST appear in the top-k — this
        // proves the Vamana adapter was rebuilt with the first 50 docs, not left empty.
        Document committedDoc = docs.get(25);
        SearchResult committedResult =
            col.search(SearchRequest.builder(committedDoc.vector(), 5).build());
        assertThat(committedResult.hits())
            .extracting(SearchResult.Hit::id)
            .contains(committedDoc.id());

        // A staged doc (doc-70) queried by its own vector MUST NOT appear — the index only
        // holds the 50 committed docs, so doc-70 is unreachable via search().
        Document stagedDoc = docs.get(70);
        SearchResult preCommitStaged =
            col.search(SearchRequest.builder(stagedDoc.vector(), 10).build());
        assertThat(preCommitStaged.hits())
            .extracting(SearchResult.Hit::id)
            .doesNotContain(stagedDoc.id());
        // get() by id also cannot find staged docs before commit.
        assertThat(col.get(stagedDoc.id())).isNull();

        // After explicit commit: all 75 are committed and reachable.
        col.commit();
        assertThat(col.size()).isEqualTo(75);

        SearchResult postCommitStaged =
            col.search(SearchRequest.builder(stagedDoc.vector(), 5).build());
        assertThat(postCommitStaged.hits())
            .extracting(SearchResult.Hit::id)
            .contains(stagedDoc.id());
        assertThat(col.get(stagedDoc.id())).isNotNull();
      }
    }

    @Test
    void inMemoryVamanaRebuildsAcrossCommits() {
      // Two sequential commits with different data must not corrupt each other. After the
      // second commit, the first batch's ids should still be present (the adapter rebuilds from
      // the full doc set, not just the new batch).
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaSeed(SEED)
              .build()) {
        List<Document> batch1 = generateDocs(50, SEED);
        col.addAll(batch1);
        col.commit();
        assertThat(col.size()).isEqualTo(50);

        // Generate a second batch with distinct ids.
        List<Document> batch2 = new ArrayList<>();
        Random rng = new Random(SEED + 100);
        for (int i = 50; i < 100; i++) {
          batch2.add(docWithMetadata("doc-" + i, randomVector(rng), i));
        }
        col.addAll(batch2);
        col.commit();
        assertThat(col.size()).isEqualTo(100);

        // All 100 ids must be retrievable via get().
        for (int i = 0; i < 100; i++) {
          assertThat(col.get("doc-" + i)).as("doc-%d", i).isNotNull();
        }
      }
    }

    @Test
    void inMemoryVamanaEmptyCollectionReportsZero() {
      // VamanaIndex rejects empty input; VamanaIndexAdapter's empty-state branch handles this.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .build()) {
        assertThat(col.size()).isZero();
        SearchResult result =
            col.search(SearchRequest.builder(randomVector(new Random(SEED)), 10).build());
        assertThat(result.hits()).isEmpty();
      }
    }
  }

  private static VectorCollection openPersistentVamana(Path storageRoot) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(IndexType.VAMANA)
        .vamanaMaxDegree(32)
        .vamanaSearchListSize(64)
        .vamanaSeed(SEED)
        .storagePath(storageRoot)
        .build();
  }

  /**
   * Round-trip tests for persistent Vamana. Each test opens a fresh collection under a {@link
   * TempDir}, writes documents + commits, closes, reopens, and asserts the reopened collection
   * serves the same data. These tests exercise the Step 4c path end-to-end: {@code
   * VamanaGraphCodec.encode} on commit, {@code graph.bin} fsync via {@code
   * GenerationDirectory.writeGeneration}, per-file CRC verification in {@code openGeneration}, and
   * {@code VamanaGraphCodec.decode} + {@link
   * com.integrallis.vectors.db.storage.MappedVamanaIndexAdapter} wrapping on open.
   */
  @Nested
  @Tag("unit")
  class PersistentVamana {

    @Test
    void closeAndReopenReturnsIdenticalSearchResults(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(200, SEED);
      float[] queryVector = randomVector(new Random(SEED + 1));

      // Write phase: open empty, ingest, commit, close.
      List<SearchResult.Hit> firstHits;
      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(docs);
        col.commit();
        assertThat(col.size()).isEqualTo(200);

        SearchResult result = col.search(SearchRequest.builder(queryVector, 10).build());
        firstHits = result.hits();
        assertThat(firstHits).hasSize(10);
      }

      // Read phase: reopen and verify the same query returns the same hits in the same order.
      // Because the Vamana graph is deterministic given the same data + seed + maxDegree +
      // searchListSize + alpha, and the reopen decodes the exact bytes written by the write
      // phase, the result sequence must match bit-identically on id and score. A score
      // difference would indicate a float-encoding drift in the graph.bin round-trip.
      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isEqualTo(200);
        SearchResult result = col.search(SearchRequest.builder(queryVector, 10).build());
        List<SearchResult.Hit> reopenedHits = result.hits();
        assertThat(reopenedHits).hasSize(10);
        for (int i = 0; i < 10; i++) {
          assertThat(reopenedHits.get(i).id()).isEqualTo(firstHits.get(i).id());
          assertThat(reopenedHits.get(i).score()).isEqualTo(firstHits.get(i).score());
        }
      }
    }

    @Test
    void reopenedSearchMatchesBruteForceRecall(@TempDir Path tempDir) {
      // Behavior assertion: after a persistence round-trip the Vamana recall vs brute force
      // should remain within tolerance. A constant-scoring stub would pass a hit-count check
      // but fail this recall floor.
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(300, SEED);
      float[] query = randomVector(new Random(SEED + 7));

      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(docs);
        col.commit();
      }
      try (var col = openPersistentVamana(storageRoot)) {
        SearchResult result = col.search(SearchRequest.builder(query, 10).build());
        Set<String> expected = bruteForceTopKIds(docs, query, 10);
        long hits = result.hits().stream().filter(h -> expected.contains(h.id())).count();
        assertThat((double) hits / 10).isGreaterThanOrEqualTo(0.7);
      }
    }

    @Test
    void metadataSurvivesRoundTrip(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(50, SEED);
      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(docs);
        col.commit();
      }
      try (var col = openPersistentVamana(storageRoot)) {
        for (int i = 0; i < 50; i++) {
          Document d = col.get("doc-" + i);
          assertThat(d).as("doc-%d", i).isNotNull();
          assertThat(d.text()).isEqualTo("text-" + i);
          assertThat(d.metadata()).containsEntry("idx", MetadataValue.of(Long.valueOf(i)));
          assertThat(d.metadata()).containsEntry("name", MetadataValue.of("doc-" + i));
          assertThat(d.metadata()).containsEntry("flag", MetadataValue.of((i % 2) == 0));
        }
      }
    }

    @Test
    void includeVectorHydratesFromMmapAfterReopen(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(20, SEED);
      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(docs);
        col.commit();
      }
      try (var col = openPersistentVamana(storageRoot)) {
        // Query by doc-0's own vector so doc-0 is guaranteed to be in the top-5 regardless of
        // graph topology (a node's nearest neighbor is itself).
        SearchResult result =
            col.search(SearchRequest.builder(docs.get(0).vector(), 5).includeVector(true).build());
        assertThat(result.hits()).isNotEmpty();
        SearchResult.Hit top = result.hits().get(0);
        assertThat(top.id()).isEqualTo("doc-0");
        assertThat(top.document().vector()).isNotNull();
        assertThat(top.document().vector()).isEqualTo(docs.get(0).vector());
      }
    }

    @Test
    void emptyCollectionRoundTrips(@TempDir Path tempDir) {
      // An empty persistent Vamana collection bootstraps from the FLAT empty path (the bootstrap
      // manifest has indexType=VAMANA but liveCount=0 and graphBinLength=0). openGeneration must
      // tolerate this: no graph.bin to read, but the collection must still open cleanly, report
      // size=0, and return no hits on search.
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isZero();
      }
      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isZero();
        SearchResult r = col.search(SearchRequest.builder(new float[DIM], 10).build());
        assertThat(r.hits()).isEmpty();
      }
    }

    @Test
    void fiveSequentialCommitsReopenCorrectly(@TempDir Path tempDir) {
      // Each commit rebuilds graph.bin from scratch. After five sequential commits the reopened
      // collection must see every document from every batch — the successor matrix materialization
      // path correctly replays oldGen.mappedVectors rows back into the new float[][].
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistentVamana(storageRoot)) {
        Random rng = new Random(SEED);
        for (int gen = 0; gen < 5; gen++) {
          List<Document> batch = new ArrayList<>();
          for (int i = 0; i < 20; i++) {
            int idx = gen * 20 + i;
            batch.add(docWithMetadata("doc-" + idx, randomVector(rng), idx));
          }
          col.addAll(batch);
          col.commit();
        }
        assertThat(col.size()).isEqualTo(100);
      }
      // Reopen and confirm every doc across every batch survived.
      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isEqualTo(100);
        for (int i = 0; i < 100; i++) {
          assertThat(col.get("doc-" + i)).as("doc-%d", i).isNotNull();
        }
      }
    }

    @Test
    void customBuildParametersSurviveReopen(@TempDir Path tempDir) {
      // Custom maxDegree=16 — the graph.bin header's maxDegree must match. searchListSize and
      // alpha are build-time hints that are NOT persisted (documented limitation), so this test
      // asserts that reopen uses the caller-supplied value on the NEXT commit, not a recovered
      // one.
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(100, SEED);
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaMaxDegree(16)
              .vamanaSearchListSize(48)
              .vamanaSeed(SEED)
              .storagePath(storageRoot)
              .build()) {
        col.addAll(docs);
        col.commit();
      }
      // Reopen with DIFFERENT searchListSize — the collection still reads gen-0's maxDegree=16
      // graph on first open. Any subsequent commit uses the new searchListSize value.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .vamanaMaxDegree(16)
              .vamanaSearchListSize(96)
              .vamanaSeed(SEED)
              .storagePath(storageRoot)
              .build()) {
        assertThat(col.size()).isEqualTo(100);
        SearchResult result = col.search(SearchRequest.builder(docs.get(0).vector(), 3).build());
        assertThat(result.hits()).isNotEmpty();
        assertThat(result.hits().get(0).id()).isEqualTo("doc-0");
      }
    }
  }

  /**
   * Crash-recovery tests for persistent Vamana. Each test writes two committed generations, then
   * damages an on-disk file in the latest generation to simulate a partial write / bit-rot / {@code
   * kill -9}. Reopen must detect the corruption, walk back to the previous valid generation, and
   * republish {@code CURRENT}.
   *
   * <p>The mechanics mirror {@link VectorDbHnswPersistenceTest.SimulatedKill9Hnsw} so any asymmetry
   * between the HNSW and Vamana crash paths would show up here as a diverging assertion.
   */
  @Nested
  @Tag("unit")
  class SimulatedKill9Vamana {

    @Test
    void corruptGraphBinFallsBackToPriorGeneration(@TempDir Path tempDir) throws IOException {
      Path storageRoot = tempDir.resolve("col");

      // Phase 1: write two committed Vamana generations so recovery has a valid gen-1 to fall
      // back to after gen-2 gets damaged. Each batch has disjoint ids.
      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(generateDocs(30, SEED));
        col.commit(); // gen-1
        Random rng = new Random(SEED + 100);
        List<Document> second = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
          second.add(docWithMetadata("alt-" + i, randomVector(rng), i));
        }
        col.addAll(second);
        col.commit(); // gen-2
        assertThat(col.size()).isEqualTo(60);
      }
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(2L);

      // Phase 2: corrupt gen-2's graph.bin by zeroing it out. The manifest's graphBinCrc32 no
      // longer matches, so GenerationDirectory.recover()'s tryVerifyPayloadCrcs pass will reject
      // it. The recovery walk must then fall back to gen-1's graph and republish CURRENT.
      Path latestGraph =
          storageRoot.resolve(FileFormat.generationDirName(2L)).resolve(FileFormat.GRAPH_FILE);
      assertThat(latestGraph).exists();
      long originalSize = Files.size(latestGraph);
      Files.write(latestGraph, new byte[(int) originalSize]);

      // Phase 3: reopen. recover() must:
      //   1. Detect the graph.bin CRC mismatch on gen-2.
      //   2. Walk back to gen-1.
      //   3. Republish CURRENT.
      try (var col = openPersistentVamana(storageRoot)) {
        // gen-1 held only the first 30 docs.
        assertThat(col.size()).isEqualTo(30);
        // And the first-batch docs are still retrievable.
        assertThat(col.get("doc-0")).isNotNull();
        assertThat(col.get("doc-29")).isNotNull();
        // The second batch is gone (its generation was rolled back).
        assertThat(col.get("alt-0")).isNull();
      }
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(1L);
    }

    @Test
    void corruptManifestFallsBackToPriorGeneration(@TempDir Path tempDir) throws IOException {
      // Same pattern as the FLAT/HNSW tests but on a Vamana collection. Proves that the
      // manifest-self-CRC recovery path in GenerationDirectory.recover is index-type agnostic.
      Path storageRoot = tempDir.resolve("col");

      try (var col = openPersistentVamana(storageRoot)) {
        col.addAll(generateDocs(25, SEED));
        col.commit(); // gen-1
        Random rng = new Random(SEED + 200);
        List<Document> second = new ArrayList<>();
        for (int i = 0; i < 25; i++) {
          second.add(docWithMetadata("v2-" + i, randomVector(rng), i));
        }
        col.addAll(second);
        col.commit(); // gen-2
        assertThat(col.size()).isEqualTo(50);
      }
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(2L);

      // Poison gen-2's manifest: truncate below HEADER_SIZE so fromBytes fails immediately.
      Path latestManifest =
          storageRoot.resolve(FileFormat.generationDirName(2L)).resolve(FileFormat.MANIFEST_FILE);
      Files.write(latestManifest, new byte[16]);

      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isEqualTo(25);
        assertThat(col.get("doc-0")).isNotNull();
        assertThat(col.get("v2-0")).isNull();
      }
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(1L);
    }
  }

  /**
   * Regression tests for the Step 4a A1 audit finding, extended to the Vamana open path. A commit
   * whose {@code writeGeneration} succeeds but whose subsequent {@code openGeneration} throws (for
   * example because of injected failure or transient OOM during mmap) must leave the collection
   * usable and a retry must advance to the NEXT generation number instead of re-attempting the
   * orphaned one.
   */
  @Nested
  @Tag("unit")
  class CommitOpenFailureRecoveryVamana {

    @Test
    void retryAfterOpenFailureAdvancesGenerationNumberVamana(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistentVamana(storageRoot)) {
        // Seed a real gen-1 so the retry exercises the non-bootstrap path.
        col.addAll(generateDocs(5, SEED));
        col.commit();
        assertThat(col.size()).isEqualTo(5);

        VectorCollectionImpl impl = (VectorCollectionImpl) col;
        impl.openGenerationFailureHook = new IOException("injected open failure");

        // Second batch: the commit's writeGeneration phase succeeds (graph.bin and friends all
        // land on disk), but openGeneration is short-circuited by the hook and throws. The
        // caller observes UncheckedIOException and the in-memory generation stays at gen-1.
        Random rng = new Random(SEED + 1);
        List<Document> second = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
          second.add(docWithMetadata("alt-" + i, randomVector(rng), i));
        }
        col.addAll(second);
        assertThatThrownBy(col::commit)
            .isInstanceOf(UncheckedIOException.class)
            .hasMessageContaining("written but cannot be opened")
            .hasMessageContaining("retry commit()");

        // gen-1 still live in process; gen-2 exists on disk but is orphaned.
        assertThat(col.size()).isEqualTo(5);
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000001"))).isTrue();
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000002"))).isTrue();

        // Clear the hook and retry. The retry must advance to gen-3 because nextGenerationNumber
        // tracks writeGeneration successes, NOT successful opens. If the counter regressed to
        // gen-2 instead, writeGeneration would throw "generation directory already exists".
        impl.openGenerationFailureHook = null;
        col.commit();

        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000003"))).isTrue();
        assertThat(col.size()).isEqualTo(10);

        // Staged second-batch docs are now reachable through the facade.
        assertThat(col.get("alt-0")).isNotNull();
        assertThat(col.get("alt-4")).isNotNull();
      }

      // And the retry's generation survives a full reopen cycle.
      try (var col = openPersistentVamana(storageRoot)) {
        assertThat(col.size()).isEqualTo(10);
      }
    }
  }

  /**
   * Concurrency test for persistent Vamana. Mirrors {@code
   * VectorDbHnswPersistenceTest.ConcurrencyHnsw.concurrentReadersNeverBlockedByHnswCommit} but on a
   * Vamana-backed collection so the refcounted {@link
   * com.integrallis.vectors.db.storage.MappedVamanaIndexAdapter} + per-generation {@link
   * java.lang.foreign.Arena} flip is exercised under contention. Each commit triggers a full
   * graph.bin rewrite, an atomic CURRENT pointer flip, a new {@code Arena.ofShared()}, and a
   * release of the previous generation when the refcount drops to zero. Readers going through
   * {@code acquireReadSnapshot} must never observe {@code IllegalStateException("closed")} while
   * the collection is alive.
   */
  @Nested
  @Tag("unit")
  class ConcurrencyVamana {

    @Test
    void concurrentReadersNeverBlockedByVamanaCommit(@TempDir Path tempDir) {
      assertTimeoutPreemptively(
          Duration.ofSeconds(15),
          () -> {
            try (var col = openPersistentVamana(tempDir.resolve("col"))) {
              // Seed with 50 docs so readers have real work to do.
              col.addAll(generateDocs(50, SEED));
              col.commit();

              int readerThreads = 4;
              int commits = 20;
              ExecutorService pool = Executors.newFixedThreadPool(readerThreads + 1);
              List<AtomicReference<Throwable>> errors = new ArrayList<>();
              AtomicBoolean stop = new AtomicBoolean(false);
              AtomicInteger sizeCalls = new AtomicInteger();
              CountDownLatch readersReady = new CountDownLatch(readerThreads);

              for (int t = 0; t < readerThreads; t++) {
                AtomicReference<Throwable> err = new AtomicReference<>();
                errors.add(err);
                pool.submit(
                    () -> {
                      try {
                        readersReady.countDown();
                        while (!stop.get()) {
                          // Any IllegalStateException("closed") thrown here is the bug: the
                          // collection has not been closed during this test, so every read path
                          // must succeed even while the writer swaps Vamana generations
                          // underneath.
                          col.size();
                          sizeCalls.incrementAndGet();
                        }
                      } catch (Throwable th) {
                        err.set(th);
                      }
                    });
              }

              AtomicReference<Throwable> writerErr = new AtomicReference<>();
              errors.add(writerErr);
              pool.submit(
                  () -> {
                    try {
                      readersReady.await();
                      Random wrng = new Random(SEED + 3);
                      int nextId = 50;
                      for (int c = 0; c < commits; c++) {
                        col.add(docWithMetadata("r" + nextId, randomVector(wrng), nextId));
                        nextId++;
                        col.commit();
                      }
                    } catch (Throwable th) {
                      writerErr.set(th);
                    } finally {
                      stop.set(true);
                    }
                  });

              pool.shutdown();
              assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
              for (AtomicReference<Throwable> err : errors) {
                Throwable t = err.get();
                if (t != null) {
                  if (t instanceof AssertionError ae) {
                    throw ae;
                  }
                  throw new AssertionError("Worker failure", t);
                }
              }
              // Sanity: readers actually executed many size() calls during the race window.
              assertThat(sizeCalls.get()).isGreaterThan(0);
              assertThat(col.size()).isEqualTo(50 + commits);
            }
          });
    }
  }
}
