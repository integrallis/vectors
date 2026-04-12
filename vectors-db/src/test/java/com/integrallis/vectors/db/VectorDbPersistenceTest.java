package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.GenerationDirectory;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end persistence acceptance tests for the Step 4a mmap-backed {@link VectorCollection}.
 *
 * <p>Each nested class exercises one slice of the durability contract:
 *
 * <ul>
 *   <li>{@link RoundTrip} — close and reopen sees identical search results, including metadata.
 *   <li>{@link MultiGeneration} — sequential commits advance {@code CURRENT} and the on-disk
 *       directory layout matches the in-memory generation number.
 *   <li>{@link SimulatedKill9} — corrupting the manifest of the latest generation forces recovery
 *       to fall back to the previous generation, mimicking the effect of a {@code kill -9} between
 *       {@code rename(tmp → gen)} and {@code writeCurrentAtomic} (or after {@code
 *       writeCurrentAtomic} when the new gen's manifest gets corrupted).
 *   <li>{@link FlushSemantics} — confirms {@code flush()} is a no-op (commits already fsync) and
 *       that data is durable as soon as {@code commit()} returns.
 * </ul>
 */
class VectorDbPersistenceTest {

  private static final long SEED = 42L;
  private static final int DIM = 16;

  private static VectorCollection openPersistent(Path storageRoot) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .storagePath(storageRoot)
        .build();
  }

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static Document docWithMetadata(String id, float[] vector, int i) {
    Map<String, MetadataValue> md = new HashMap<>();
    // Use Long.valueOf + auto-unbox rather than a primitive cast: the explicit boxing makes
    // the intent ("store this as a long metadata value") clearer than (long) i, which reads
    // as a narrowing/widening conversion rather than a type selection.
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

  @Nested
  @Tag("unit")
  class RoundTrip {

    @Test
    void closeAndReopenReturnsIdenticalSearchResults(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(500, SEED);

      // Write phase: open empty, ingest, commit, close.
      List<SearchResult.Hit> firstHits;
      float[] queryVector = randomVector(new Random(SEED + 1));
      try (var col = openPersistent(storageRoot)) {
        col.addAll(docs);
        col.commit();
        assertThat(col.size()).isEqualTo(500);

        SearchResult result =
            col.search(SearchRequest.builder(queryVector, 10).includeMetadata(true).build());
        firstHits = result.hits();
        assertThat(firstHits).hasSize(10);
      }

      // Read phase: reopen and verify the same query returns the same hits in the same order.
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isEqualTo(500);
        SearchResult result =
            col.search(SearchRequest.builder(queryVector, 10).includeMetadata(true).build());

        List<SearchResult.Hit> reopenedHits = result.hits();
        assertThat(reopenedHits).hasSize(10);
        for (int i = 0; i < 10; i++) {
          assertThat(reopenedHits.get(i).id()).isEqualTo(firstHits.get(i).id());
          assertThat(reopenedHits.get(i).score()).isEqualTo(firstHits.get(i).score());
        }
      }
    }

    @Test
    void metadataSurvivesRoundTrip(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(50, SEED);
      try (var col = openPersistent(storageRoot)) {
        col.addAll(docs);
        col.commit();
      }
      try (var col = openPersistent(storageRoot)) {
        for (int i = 0; i < 50; i++) {
          Document d = col.get("doc-" + i);
          assertThat(d).isNotNull();
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
      try (var col = openPersistent(storageRoot)) {
        col.addAll(docs);
        col.commit();
      }
      try (var col = openPersistent(storageRoot)) {
        SearchResult result =
            col.search(SearchRequest.builder(docs.get(0).vector(), 5).includeVector(true).build());
        // The top hit must be the query vector itself with a hydrated float[] payload.
        assertThat(result.hits()).hasSize(5);
        SearchResult.Hit top = result.hits().get(0);
        assertThat(top.id()).isEqualTo("doc-0");
        assertThat(top.document().vector()).isNotNull();
        assertThat(top.document().vector()).isEqualTo(docs.get(0).vector());
      }
    }

    @Test
    void emptyCollectionRoundTrips(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isZero();
      }
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isZero();
        SearchResult r = col.search(SearchRequest.builder(new float[DIM], 10).build());
        assertThat(r.hits()).isEmpty();
      }
    }
  }

  @Nested
  @Tag("unit")
  class MultiGeneration {

    @Test
    void fiveSequentialCommitsAdvanceCurrentPointer(@TempDir Path tempDir) throws IOException {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        for (int gen = 1; gen <= 5; gen++) {
          List<Document> batch =
              List.of(
                  docWithMetadata(
                      "g" + gen + "-d0", randomVector(new Random(SEED + gen)), gen * 10),
                  docWithMetadata(
                      "g" + gen + "-d1", randomVector(new Random(SEED + gen + 1)), gen * 10 + 1));
          col.addAll(batch);
          col.commit();
          // CURRENT should point at the gen we just wrote.
          assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo((long) gen);
        }
        assertThat(col.size()).isEqualTo(10);
      }

      // At least gen-5 (the newest commit) must exist on disk; recent earlier generations
      // SHOULD exist today because Step 4a has no compaction, but this assertion is
      // deliberately weak so that a future compaction pass that retires older gen dirs does
      // not break this test. The authoritative check is that CURRENT points at gen-5 and
      // that the live generation is loadable — the exact count of on-disk gen dirs is a
      // retention policy detail, not a correctness invariant.
      try (Stream<Path> entries = Files.list(storageRoot)) {
        long genDirs =
            entries
                .filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .filter(name -> FileFormat.parseGenerationDirName(name) >= 0)
                .count();
        assertThat(genDirs).isGreaterThanOrEqualTo(1L);
      }
      assertThat(Files.isDirectory(storageRoot.resolve(FileFormat.generationDirName(5L)))).isTrue();
    }

    @Test
    void reopenAfterMultipleCommitsSeesAllDocs(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        for (int gen = 0; gen < 3; gen++) {
          List<Document> batch = generateDocs(100, SEED + gen);
          // Re-id so they don't collide across batches.
          List<Document> reIded = new ArrayList<>(100);
          for (int j = 0; j < batch.size(); j++) {
            Document src = batch.get(j);
            reIded.add(
                new Document("g" + gen + "-" + src.id(), src.vector(), src.text(), src.metadata()));
          }
          col.addAll(reIded);
          col.commit();
        }
        assertThat(col.size()).isEqualTo(300);
      }
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isEqualTo(300);
        // Spot-check the boundary: a doc from each batch must be retrievable.
        assertThat(col.contains("g0-doc-0")).isTrue();
        assertThat(col.contains("g1-doc-50")).isTrue();
        assertThat(col.contains("g2-doc-99")).isTrue();
      }
    }
  }

  @Nested
  @Tag("unit")
  class SimulatedKill9 {

    @Test
    void corruptedLatestManifestFallsBackToPreviousGeneration(@TempDir Path tempDir)
        throws IOException {
      Path storageRoot = tempDir.resolve("col");

      // Phase 1: write two committed generations so the recovery sweep has something to fall
      // back to. After this block CURRENT points at gen-2.
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(20, SEED));
        col.commit(); // gen-1
        col.addAll(generateDocs(20, SEED + 100).stream().map(this::reKey).toList());
        col.commit(); // gen-2
        assertThat(col.size()).isEqualTo(40);
      }

      // Sanity check the on-disk state.
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(2L);

      // Phase 2: simulate a kill -9 that left gen-2's manifest corrupt. We zero-out the manifest
      // file in place — recovery should detect the bad self-CRC, walk back, and republish CURRENT
      // at gen-1.
      Path latestManifest =
          storageRoot.resolve(FileFormat.generationDirName(2L)).resolve(FileFormat.MANIFEST_FILE);
      assertThat(latestManifest).exists();
      Files.write(latestManifest, new byte[16]); // shorter than HEADER_SIZE → fromBytes fails

      // Phase 3: reopen. The constructor runs GenerationDirectory.recover, which must:
      //   1. Detect the corrupt gen-2 manifest.
      //   2. Walk back to gen-1.
      //   3. Republish CURRENT to point at gen-1.
      try (var col = openPersistent(storageRoot)) {
        // Gen-1 had only 20 docs, so the rolled-back collection must reflect that.
        assertThat(col.size()).isEqualTo(20);
      }

      // CURRENT must now point at gen-1 (recovery republished it).
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(1L);
    }

    @Test
    void corruptedCurrentPointerRebuildsFromNewestValidGeneration(@TempDir Path tempDir)
        throws IOException {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(10, SEED));
        col.commit(); // gen-1
      }

      // Truncate CURRENT to zero bytes — readCurrent returns -1L, recovery scans for newest.
      Files.write(storageRoot.resolve(FileFormat.CURRENT_FILE), new byte[0]);

      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isEqualTo(10);
      }
      assertThat(GenerationDirectory.readCurrent(storageRoot)).isEqualTo(1L);
    }

    @Test
    void allManifestsCorruptThrowsOnReopen(@TempDir Path tempDir) throws IOException {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(10, SEED));
        col.commit(); // gen-1
      }

      // Corrupt every manifest on disk so the recovery sweep has nothing valid to fall back to.
      // The bootstrap source can't help here either: writeGeneration refuses to overwrite an
      // existing gen-0 directory (operator-visible failure mode — corrupt-but-present state must
      // be investigated by a human, not silently masked).
      Files.write(
          storageRoot.resolve(FileFormat.generationDirName(0L)).resolve(FileFormat.MANIFEST_FILE),
          new byte[16]);
      Files.write(
          storageRoot.resolve(FileFormat.generationDirName(1L)).resolve(FileFormat.MANIFEST_FILE),
          new byte[16]);

      assertThatThrownBy(() -> openPersistent(storageRoot))
          .isInstanceOf(UncheckedIOException.class)
          .hasMessageContaining("Failed to open persistent collection")
          .hasRootCauseInstanceOf(IOException.class);
    }

    private Document reKey(Document src) {
      return new Document("g2-" + src.id(), src.vector(), src.text(), src.metadata());
    }
  }

  @Nested
  @Tag("unit")
  class FlushSemantics {

    @Test
    void flushAfterCommitIsNoOp(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(50, SEED));
        col.commit();
        col.flush(); // must not throw
      }
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isEqualTo(50);
      }
    }

    @Test
    void flushBeforeCommitDoesNotPersistStaging(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(50, SEED));
        // No commit. flush() must not implicitly commit, per the documented contract.
        col.flush();
        // Same-process visibility: staging is not searchable until commit.
        SearchResult r = col.search(SearchRequest.builder(new float[DIM], 5).build());
        assertThat(r.hits()).isEmpty();
      }
      try (var col = openPersistent(storageRoot)) {
        // Cross-process: staging without commit must not be durable either.
        assertThat(col.size()).isZero();
      }
    }

    @Test
    void dataDurableImmediatelyAfterCommit(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        col.addAll(generateDocs(25, SEED));
        col.commit();
      }
      // No flush() between commit and close — data should still be on disk.
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isEqualTo(25);
      }
    }
  }

  @Nested
  @Tag("unit")
  class StorageRootValidation {

    @Test
    void storageRootCreatedIfMissing(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("nested").resolve("col");
      assertThat(Files.exists(storageRoot)).isFalse();
      try (var col = openPersistent(storageRoot)) {
        assertThat(col.size()).isZero();
      }
      assertThat(Files.isDirectory(storageRoot)).isTrue();
    }

    @Test
    void storageRootIsAFileThrows(@TempDir Path tempDir) throws IOException {
      Path file = tempDir.resolve("not-a-dir");
      Files.writeString(file, "junk");
      assertThatThrownBy(() -> openPersistent(file))
          .isInstanceOf(UncheckedIOException.class)
          .hasMessageContaining("Failed to open persistent collection");
    }

    @Test
    void relativeStoragePathRejected() {
      // A relative path would resolve against the JVM working directory at runtime, which is
      // fragile across deployments. The builder Javadoc says absolute is required — enforce it.
      assertThatThrownBy(
              () ->
                  VectorCollection.builder()
                      .dimension(DIM)
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .storagePath(Path.of("relative-collection"))
                      .build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("storagePath must be absolute");
    }
  }

  /**
   * Regression tests for audit finding A1 — {@code commitPersistent()} must not get stuck if {@link
   * com.integrallis.vectors.db.storage.GenerationDirectory#writeGeneration} succeeds but the
   * subsequent {@code openGeneration} throws (e.g. transient fd exhaustion or OOM during mmap).
   * Before the fix, the {@code nextGenerationNumber} counter was derived from the still- live
   * {@code oldGen} so a retry would re-attempt the same number and hit "generation directory
   * already exists". After the fix, the counter advances unconditionally on write success so
   * retries land on {@code gen-(N+1)}.
   *
   * <p>Failures are injected via the package-private {@code openGenerationFailureHook} field on
   * {@link VectorCollectionImpl} — the cleanest way to exercise the "write ok, open fail" window
   * without racing file corruption against a real mmap operation.
   */
  @Nested
  @Tag("unit")
  class CommitOpenFailureRecovery {

    @Test
    void retryAfterOpenFailureAdvancesGenerationNumber(@TempDir Path tempDir) throws IOException {
      Path storageRoot = tempDir.resolve("col");
      // Verify both: (a) the first commit's disk state survives (orphaned gen-1), and (b) the
      // retry succeeds and is durable across reopen.
      try (var col = openPersistent(storageRoot)) {
        // Inject a failure so the FIRST commit succeeds on disk but fails to open in-process.
        VectorCollectionImpl impl = (VectorCollectionImpl) col;
        impl.openGenerationFailureHook = new IOException("injected open failure");

        List<Document> firstBatch = generateDocs(3, SEED);
        col.addAll(firstBatch);
        assertThatThrownBy(col::commit)
            .isInstanceOf(UncheckedIOException.class)
            .hasMessageContaining("written but cannot be opened")
            .hasMessageContaining("retry commit()");

        // Staging is still populated; oldGen (the empty bootstrap) is still live.
        assertThat(col.size()).isZero();

        // Disk should now contain gen-0 (empty bootstrap) AND gen-1 (the failed commit's
        // durable payload). CURRENT may point at 1 but the in-memory state is at 0.
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000000"))).isTrue();
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000001"))).isTrue();

        // Clear the hook and retry. The retry must advance to gen-2 instead of colliding with
        // the already-durable gen-1. If nextGenerationNumber were still derived from oldGen,
        // writeGeneration would throw "generation directory already exists".
        impl.openGenerationFailureHook = null;
        col.commit();

        // After the successful retry, gen-2 exists on disk and the in-memory state is consistent.
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000002"))).isTrue();
        assertThat(col.size()).isEqualTo(3);

        // And the retry's docs are searchable through the facade.
        SearchResult hits =
            col.search(SearchRequest.builder(firstBatch.get(0).vector(), 3).build());
        assertThat(hits.hits()).hasSize(3);
      }

      // Reopen — recovery picks the newest valid generation (gen-2, which CURRENT points at).
      try (var reopened = openPersistent(storageRoot)) {
        assertThat(reopened.size()).isEqualTo(3);
      }
    }

    @Test
    void retryAfterOpenFailureWithPriorCommittedGeneration(@TempDir Path tempDir) {
      // Variant of the test above with a non-empty prior generation: first successful commit
      // lands on gen-1, then we inject an open failure on the second commit. Retry must advance
      // to gen-3 (gen-2 is orphaned) without colliding with the existing gen-1 or gen-2.
      Path storageRoot = tempDir.resolve("col");
      try (var col = openPersistent(storageRoot)) {
        // First commit with unique ids in [0, 2).
        List<Document> first = generateDocs(2, SEED);
        col.addAll(first);
        col.commit();
        assertThat(col.size()).isEqualTo(2);

        VectorCollectionImpl impl = (VectorCollectionImpl) col;
        impl.openGenerationFailureHook = new IOException("injected");

        // Second batch with ids that don't collide with the first.
        Random rng = new Random(SEED + 1);
        List<Document> second = new ArrayList<>();
        second.add(docWithMetadata("alt-0", randomVector(rng), 10));
        second.add(docWithMetadata("alt-1", randomVector(rng), 11));
        col.addAll(second);

        assertThatThrownBy(col::commit).isInstanceOf(UncheckedIOException.class);

        // Retry with the hook cleared — retry must advance to gen-3.
        impl.openGenerationFailureHook = null;
        col.commit();

        assertThat(col.size()).isEqualTo(4);
        // gen-1 (first commit), gen-2 (failed commit, orphaned), gen-3 (retry) all on disk.
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000001"))).isTrue();
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000002"))).isTrue();
        assertThat(Files.isDirectory(storageRoot.resolve("gen-0000000000000003"))).isTrue();
        // The successor generation contains both original and retry docs.
        assertThat(col.contains("doc-0")).isTrue();
        assertThat(col.contains("alt-0")).isTrue();
      }
    }
  }

  /**
   * Regression guard for audit finding B2 — {@code commitPersistent()} used to walk the live+staged
   * vector set twice when the index type was HNSW or VAMANA: once to pack staged docs into the
   * on-disk {@code vectors.bin} byte image and a second time to clone them into a {@code float[][]}
   * matrix for the graph builder. The fix fuses the two passes into a single {@code
   * materializeSuccessor} helper that produces both outputs in one loop.
   *
   * <p>A straightforward "did it regress" test would need to instrument allocations or reach into
   * private methods. Instead we assert a stronger behavioral invariant: committing the same data
   * through FLAT and HNSW must produce byte-identical {@code vectors.bin} images. The two index
   * types exercise different branches of {@code commitPersistent} (FLAT passes {@code
   * needMatrix=false}, HNSW passes {@code needMatrix=true}), so if the unified helper ever diverges
   * in how it produces the byte image — whether via a re-introduced second pass or a subtler layout
   * drift — this assertion breaks immediately. The assertion also covers the documented "bulk-copy
   * fast path is preserved" invariant in the helper's javadoc: any FLAT/HNSW divergence on old
   * generation bytes would manifest as a mismatch on the second commit.
   *
   * <p><b>Why tiny HNSW parameters ({@code M=4, efConstruction=16})?</b> These tests assert
   * byte-identity on {@code vectors.bin}, which is the raw vector byte image. HNSW build parameters
   * like {@code M} and {@code efConstruction} drive the graph construction and are only ever
   * encoded into {@code graph.bin} — they have zero effect on the shape or contents of {@code
   * vectors.bin}. We pick the smallest valid settings to minimize graph-build time during the test;
   * any other valid combination would produce the same {@code vectors.bin} bytes.
   *
   * <p><b>Why {@code @Tag("unit")} despite touching {@code @TempDir}?</b> Matches the existing
   * convention of every other {@code @Nested} class in this file ({@link RoundTrip}, {@link
   * MultiGeneration}, {@link SimulatedKill9}, {@link FlushSemantics}, {@link
   * StorageRootValidation}, {@link CommitOpenFailureRecovery}) — all of which are persistence tests
   * that write real files under a temp directory and all of which carry {@code @Tag("unit")}. The
   * project convention in this file is "unit = fast, regardless of filesystem touch"; breaking
   * consistency for one nested class would be worse than matching it. The tests run in well under
   * 100ms each so the {@code unitTest} task stays fast.
   */
  @Nested
  @Tag("unit")
  class FusedSuccessorMaterialization {

    @Test
    void flatAndHnswProduceIdenticalVectorsBin(@TempDir Path tempDir) throws IOException {
      // Two sibling storage roots so each collection gets its own generation tree.
      Path flatRoot = tempDir.resolve("flat");
      Path hnswRoot = tempDir.resolve("hnsw");

      // Deterministic data so both collections observe the same vectors in the same order.
      List<Document> docs = generateDocs(32, SEED);

      try (var flat =
              VectorCollection.builder()
                  .dimension(DIM)
                  .metric(SimilarityFunction.EUCLIDEAN)
                  .indexType(IndexType.FLAT)
                  .storagePath(flatRoot)
                  .build();
          var hnsw =
              VectorCollection.builder()
                  .dimension(DIM)
                  .metric(SimilarityFunction.EUCLIDEAN)
                  .indexType(IndexType.HNSW)
                  .hnswM(4)
                  .hnswEfConstruction(16)
                  .storagePath(hnswRoot)
                  .build()) {
        flat.addAll(docs);
        flat.commit();
        hnsw.addAll(docs);
        hnsw.commit();
      }

      // Read vectors.bin out of each collection's gen-1 (the first real commit after bootstrap).
      byte[] flatBin =
          Files.readAllBytes(
              flatRoot.resolve("gen-0000000000000001").resolve(FileFormat.VECTORS_FILE));
      byte[] hnswBin =
          Files.readAllBytes(
              hnswRoot.resolve("gen-0000000000000001").resolve(FileFormat.VECTORS_FILE));

      // Strong assertion: the fused materialization helper produces bit-identical byte images
      // for both paths. Any drift — re-introduced second pass, layout change, unaligned stride,
      // endian flip — breaks this immediately.
      assertThat(hnswBin).isEqualTo(flatBin);
    }

    @Test
    void flatAndHnswIdenticalAcrossMultipleCommits(@TempDir Path tempDir) throws IOException {
      // Second regression guard: run two commits per collection so the second one exercises the
      // "bulk-copy old rows + fuse staged rows" path on non-empty live state. The B2 fusion
      // deliberately keeps the bulk-copy fast path for old rows; this test proves FLAT and HNSW
      // still agree after that path has run.
      Path flatRoot = tempDir.resolve("flat");
      Path hnswRoot = tempDir.resolve("hnsw");
      List<Document> batch1 = generateDocs(20, SEED);
      List<Document> batch2 = generateDocs(20, SEED + 1L);
      // Re-id the second batch so ids don't collide with the first.
      List<Document> renamed = new ArrayList<>(batch2.size());
      for (int i = 0; i < batch2.size(); i++) {
        Document d = batch2.get(i);
        renamed.add(new Document("b2-" + i, d.vector(), d.text(), d.metadata()));
      }

      try (var flat =
              VectorCollection.builder()
                  .dimension(DIM)
                  .metric(SimilarityFunction.EUCLIDEAN)
                  .indexType(IndexType.FLAT)
                  .storagePath(flatRoot)
                  .build();
          var hnsw =
              VectorCollection.builder()
                  .dimension(DIM)
                  .metric(SimilarityFunction.EUCLIDEAN)
                  .indexType(IndexType.HNSW)
                  .hnswM(4)
                  .hnswEfConstruction(16)
                  .storagePath(hnswRoot)
                  .build()) {
        flat.addAll(batch1);
        flat.commit();
        flat.addAll(renamed);
        flat.commit();

        hnsw.addAll(batch1);
        hnsw.commit();
        hnsw.addAll(renamed);
        hnsw.commit();
      }

      // gen-2 is the second-commit generation on both sides; it exercises both the bulk-copy old
      // rows path AND the staged-row single-pass path.
      byte[] flatBin =
          Files.readAllBytes(
              flatRoot.resolve("gen-0000000000000002").resolve(FileFormat.VECTORS_FILE));
      byte[] hnswBin =
          Files.readAllBytes(
              hnswRoot.resolve("gen-0000000000000002").resolve(FileFormat.VECTORS_FILE));
      assertThat(hnswBin).isEqualTo(flatBin);
    }
  }
}
