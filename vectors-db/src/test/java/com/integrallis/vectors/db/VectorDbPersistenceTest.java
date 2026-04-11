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
    md.put("idx", MetadataValue.of((long) i));
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
          assertThat(d.metadata()).containsEntry("idx", MetadataValue.of((long) i));
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

      // Five gen dirs (1..5) plus the bootstrap gen-0 must exist on disk.
      try (Stream<Path> entries = Files.list(storageRoot)) {
        long genDirs =
            entries
                .filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .filter(name -> FileFormat.parseGenerationDirName(name) >= 0)
                .count();
        assertThat(genDirs).isEqualTo(6L); // gen-0 bootstrap + 5 commits
      }
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
  }
}
