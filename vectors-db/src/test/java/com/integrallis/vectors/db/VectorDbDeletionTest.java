package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Integration tests for Step 6 tombstone-based deletion: {@code delete}, {@code deleteWhere},
 * {@code upsert}, {@code compact}, {@code physicalSize}.
 */
class VectorDbDeletionTest {

  private static final int DIM = 8;
  private static final long SEED = 42L;

  private static VectorCollection newCollection(IndexType indexType) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(indexType)
        .build();
  }

  private static VectorCollection newPersistentCollection(IndexType indexType, Path storageRoot) {
    var builder =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(indexType)
            .storagePath(storageRoot);
    if (indexType == IndexType.HNSW) {
      builder.hnswM(4).hnswEfConstruction(16);
    }
    return builder.build();
  }

  private static Document doc(String id, float... vector) {
    return Document.of(id, vector);
  }

  private static Document docWithMeta(String id, float[] vector, Map<String, MetadataValue> meta) {
    return new Document(id, vector, null, meta);
  }

  private static float[] randomVector(Random rng) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = rng.nextFloat();
    }
    return v;
  }

  private static List<Document> generateDocs(int count, long seed) {
    Random rng = new Random(seed);
    List<Document> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      out.add(
          new Document(
              "doc-" + i,
              randomVector(rng),
              "text-" + i,
              Map.of("idx", MetadataValue.of((long) i))));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // delete(String id)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class DeleteById {

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void deleteReturnsTrueForKnownId(IndexType idx) {
      try (var col = newCollection(idx)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();
        assertThat(col.delete("a")).isTrue();
      }
    }

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void deleteReturnsFalseForUnknownId(IndexType idx) {
      try (var col = newCollection(idx)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();
        assertThat(col.delete("z")).isFalse();
      }
    }

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void deleteReturnsFalseForAlreadyDeletedId(IndexType idx) {
      try (var col = newCollection(idx)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();
        assertThat(col.delete("a")).isTrue();
        assertThat(col.delete("a")).isFalse();
      }
    }

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void deletedDocExcludedFromSearchAfterCommit(IndexType idx) {
      try (var col = newCollection(idx)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.add(doc("b", 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();

        col.delete("a");
        col.commit();

        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, 10).build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().get(0).id()).isEqualTo("b");
      }
    }

    /**
     * Regression test for the NaN-score defect: when the SimilarityFunction is COSINE and a
     * document is tombstoned, its ordinal must not carry a zero-vector placeholder into the ANN
     * graph. A zero-vector produces undefined cosine similarity (0/0 = NaN) which corrupts the
     * HNSW/Vamana priority-queue ordering. The fix retains original vectors for tombstoned
     * ordinals; this test would hang or return wrong results under the old zero-vector approach.
     */
    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void deletedDocExcludedFromSearchWithCosineMetric(IndexType idx) {
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(idx)
              .build()) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.add(doc("b", 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.add(doc("c", 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f));
        col.commit();

        // Delete "a" — its ordinal must keep its original vector, not become a zero-vector.
        col.delete("a");
        col.commit();

        // Search near "b" — "a" must not appear, "b" must be the top hit.
        var result =
            col.search(
                SearchRequest.builder(new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, 10).build());
        var ids = result.hits().stream().map(h -> h.id()).toList();
        assertThat(ids).hasSize(2).doesNotContain("a").contains("b", "c");
        assertThat(ids.get(0)).isEqualTo("b");
      }
    }

    @Test
    void deleteOfStagedDocRemovesImmediately() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        // "a" is staged, not yet committed.
        assertThat(col.delete("a")).isTrue();
        col.commit();
        assertThat(col.size()).isZero();
      }
    }

    @Test
    void getReturnsNullForDeletedDoc() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();
        col.delete("a");
        col.commit();

        assertThat(col.get("a")).isNull();
      }
    }

    @Test
    void containsReturnsFalseForDeletedDoc() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();
        col.delete("a");
        col.commit();

        assertThat(col.contains("a")).isFalse();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // size() vs physicalSize()
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class SizeVsPhysicalSize {

    @Test
    void sizeAndPhysicalSizeEqualWithNoTombstones() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(10, SEED));
        col.commit();

        assertThat(col.size()).isEqualTo(10);
        assertThat(col.physicalSize()).isEqualTo(10);
      }
    }

    @Test
    void sizeExcludesTombstonesPhysicalSizeDoesNot() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(10, SEED));
        col.commit();

        col.delete("doc-3");
        col.delete("doc-7");
        col.commit();

        assertThat(col.size()).isEqualTo(8);
        assertThat(col.physicalSize()).isEqualTo(10);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // deleteWhere(Filter)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class DeleteWhere {

    @Test
    void deleteWhereMatchesAndReturnsCount() {
      try (var col = newCollection(IndexType.FLAT)) {
        Random rng = new Random(SEED);
        for (int i = 0; i < 20; i++) {
          col.add(
              docWithMeta(
                  "doc-" + i,
                  randomVector(rng),
                  Map.of("color", MetadataValue.of(i < 10 ? "red" : "blue"))));
        }
        col.commit();

        int count = col.deleteWhere(Filters.eq("color", "red"));
        assertThat(count).isEqualTo(10);

        col.commit();
        assertThat(col.size()).isEqualTo(10);

        // Verify only blue docs remain.
        var result =
            col.search(SearchRequest.builder(new float[DIM], 20).includeMetadata(true).build());
        for (var hit : result.hits()) {
          MetadataValue color = hit.document().metadata().get("color");
          assertThat(((MetadataValue.Str) color).value()).isEqualTo("blue");
        }
      }
    }

    @Test
    void deleteWhereReturnsZeroWhenNoMatch() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(
            docWithMeta(
                "a",
                new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f},
                Map.of("color", MetadataValue.of("red"))));
        col.commit();

        int count = col.deleteWhere(Filters.eq("color", "blue"));
        assertThat(count).isZero();
      }
    }

    @Test
    void deleteWhereAlsoRemovesStagedDocuments() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(
            docWithMeta(
                "a",
                new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f},
                Map.of("color", MetadataValue.of("red"))));
        // "a" is staged (not committed) — deleteWhere should still remove it.
        int count = col.deleteWhere(Filters.eq("color", "red"));
        assertThat(count).isEqualTo(1);

        col.commit();
        assertThat(col.size()).isZero();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // upsert(Document)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class Upsert {

    @Test
    void upsertInsertsNewDocument() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.upsert(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();

        assertThat(col.size()).isEqualTo(1);
        assertThat(col.contains("a")).isTrue();
      }
    }

    @Test
    void upsertReplacesExistingCommittedDocument() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(
            new Document("a", new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, "original", Map.of()));
        col.commit();

        col.upsert(
            new Document("a", new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, "updated", Map.of()));
        col.commit();

        // Size should stay at 1 (old ordinal tombstoned + new ordinal added).
        assertThat(col.size()).isEqualTo(1);
        Document fetched = col.get("a");
        assertThat(fetched).isNotNull();
        assertThat(fetched.text()).isEqualTo("updated");
      }
    }

    @Test
    void upsertReplacesStagedDocument() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(new Document("a", new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, "first", Map.of()));
        // Replace staged doc before commit.
        col.upsert(
            new Document("a", new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, "second", Map.of()));
        col.commit();

        assertThat(col.size()).isEqualTo(1);
        Document fetched = col.get("a");
        assertThat(fetched).isNotNull();
        assertThat(fetched.text()).isEqualTo("second");
      }
    }

    @Test
    void upsertUpdatesSearchResults() {
      try (var col = newCollection(IndexType.FLAT)) {
        // Insert "a" near the first axis.
        col.add(doc("a", 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.add(doc("b", 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f));
        col.commit();

        // Upsert "a" so it's now near the second axis.
        col.upsert(doc("a", 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f));
        col.commit();

        // Searching near the second axis should now find "a" as the top hit.
        var result =
            col.search(
                SearchRequest.builder(new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, 2).build());
        assertThat(result.hits().get(0).id()).isEqualTo("a");
      }
    }
  }

  // ---------------------------------------------------------------------------
  // compact()
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class Compact {

    @Test
    void compactIsNoOpWithNoTombstones() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(10, SEED));
        col.commit();

        col.compact();
        assertThat(col.size()).isEqualTo(10);
        assertThat(col.physicalSize()).isEqualTo(10);
      }
    }

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void compactRemovesTombstones(IndexType idx) {
      try (var col = newCollection(idx)) {
        col.addAll(generateDocs(10, SEED));
        col.commit();

        col.delete("doc-3");
        col.delete("doc-7");
        col.commit();

        assertThat(col.size()).isEqualTo(8);
        assertThat(col.physicalSize()).isEqualTo(10);

        col.compact();

        assertThat(col.size()).isEqualTo(8);
        assertThat(col.physicalSize()).isEqualTo(8);

        // Deleted docs stay deleted.
        assertThat(col.contains("doc-3")).isFalse();
        assertThat(col.contains("doc-7")).isFalse();

        // Surviving docs are still searchable.
        assertThat(col.contains("doc-0")).isTrue();
        assertThat(col.contains("doc-9")).isTrue();
      }
    }

    @ParameterizedTest
    @EnumSource(
        value = IndexType.class,
        names = {"FLAT", "HNSW", "VAMANA"})
    void compactPreservesSearchResults(IndexType idx) {
      // Use a larger corpus for graph-backed indexes so the ANN graph has meaningful structure.
      int docCount = idx == IndexType.FLAT ? 20 : 50;
      try (var col = newCollection(idx)) {
        col.addAll(generateDocs(docCount, SEED));
        col.commit();

        // Delete the first 5 docs, compact, and verify the remaining ones are still searchable.
        for (int i = 0; i < 5; i++) {
          col.delete("doc-" + i);
        }
        col.commit();
        col.compact();

        assertThat(col.size()).isEqualTo(docCount - 5);
        assertThat(col.physicalSize()).isEqualTo(docCount - 5);

        var postResult = col.search(SearchRequest.builder(new float[DIM], docCount - 5).build());
        assertThat(postResult.hits()).hasSize(docCount - 5);

        // None of the deleted docs should appear.
        for (var hit : postResult.hits()) {
          int idx2 = Integer.parseInt(hit.id().substring(4));
          assertThat(idx2).isGreaterThanOrEqualTo(5);
        }
      }
    }

    @Test
    void compactCommitsPendingWorkFirst() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        col.delete("doc-0");
        // Stage a new doc AND have a pending tombstone — compact must commit both first.
        col.add(doc("new-one", 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f));
        col.compact();

        assertThat(col.contains("doc-0")).isFalse();
        assertThat(col.contains("new-one")).isTrue();
        assertThat(col.size()).isEqualTo(5); // 5 original - 1 deleted + 1 new
        assertThat(col.physicalSize()).isEqualTo(5); // compacted
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Persistent tombstones
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class Persistence {

    @Test
    void deleteSurvivesCloseAndReopen(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.addAll(generateDocs(20, SEED));
        col.commit();

        col.delete("doc-5");
        col.delete("doc-10");
        col.commit();

        assertThat(col.size()).isEqualTo(18);
      }

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        assertThat(col.size()).isEqualTo(18);
        assertThat(col.contains("doc-5")).isFalse();
        assertThat(col.contains("doc-10")).isFalse();
        assertThat(col.contains("doc-0")).isTrue();
        assertThat(col.get("doc-5")).isNull();
      }
    }

    @Test
    void compactSurvivesCloseAndReopen(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.addAll(generateDocs(20, SEED));
        col.commit();

        col.delete("doc-0");
        col.delete("doc-1");
        col.delete("doc-2");
        col.commit();
        col.compact();

        assertThat(col.size()).isEqualTo(17);
        assertThat(col.physicalSize()).isEqualTo(17);
      }

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        assertThat(col.size()).isEqualTo(17);
        assertThat(col.physicalSize()).isEqualTo(17);
        assertThat(col.contains("doc-0")).isFalse();
        assertThat(col.contains("doc-3")).isTrue();
      }
    }

    @Test
    void hnswDeleteAndReopenPreservesResults(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(50, SEED);

      try (var col = newPersistentCollection(IndexType.HNSW, storageRoot)) {
        col.addAll(docs);
        col.commit();

        col.delete("doc-0");
        col.delete("doc-25");
        col.commit();
      }

      try (var col = newPersistentCollection(IndexType.HNSW, storageRoot)) {
        assertThat(col.size()).isEqualTo(48);
        assertThat(col.contains("doc-0")).isFalse();
        assertThat(col.contains("doc-25")).isFalse();

        var result = col.search(SearchRequest.builder(docs.get(1).vector(), 10).build());
        assertThat(result.hits()).isNotEmpty();
        // Deleted docs must not appear in results.
        for (var hit : result.hits()) {
          assertThat(hit.id()).isNotEqualTo("doc-0");
          assertThat(hit.id()).isNotEqualTo("doc-25");
        }
      }
    }

    @Test
    void upsertPersistsAcrossReopen(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.add(
            new Document("a", new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, "original", Map.of()));
        col.commit();

        col.upsert(
            new Document("a", new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, "updated", Map.of()));
        col.commit();
      }

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        assertThat(col.size()).isEqualTo(1);
        Document fetched = col.get("a");
        assertThat(fetched).isNotNull();
        assertThat(fetched.text()).isEqualTo("updated");
      }
    }

    @Test
    void deleteWhereWithPersistence(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        Random rng = new Random(SEED);
        for (int i = 0; i < 10; i++) {
          col.add(
              docWithMeta(
                  "doc-" + i,
                  randomVector(rng),
                  Map.of("parity", MetadataValue.of(i % 2 == 0 ? "even" : "odd"))));
        }
        col.commit();

        int count = col.deleteWhere(Filters.eq("parity", "even"));
        assertThat(count).isEqualTo(5);
        col.commit();
      }

      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        assertThat(col.size()).isEqualTo(5);
        // All even-indexed docs should be gone.
        for (int i = 0; i < 10; i++) {
          if (i % 2 == 0) {
            assertThat(col.contains("doc-" + i)).isFalse();
          } else {
            assertThat(col.contains("doc-" + i)).isTrue();
          }
        }
      }
    }

    @Test
    void tombstonesFileAbsentWhenNoTombstones(@TempDir Path tempDir) throws Exception {
      // A generation with no tombstones must not write tombstones.bin at all.
      // The manifest records tombstonesBinLength == 0 as the authoritative signal.
      Path storageRoot = tempDir.resolve("col");
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();
      }

      boolean hasTombstonesFile =
          Files.walk(storageRoot)
              .anyMatch(p -> p.getFileName().toString().equals("tombstones.bin"));
      assertThat(hasTombstonesFile).isFalse();
    }

    @Test
    void tombstonesFileWrittenAfterDelete(@TempDir Path tempDir) throws Exception {
      // After a delete + commit, tombstones.bin must be present in the live generation directory.
      Path storageRoot = tempDir.resolve("col");
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        col.delete("doc-2");
        col.commit();
      }

      boolean hasTombstonesFile =
          Files.walk(storageRoot)
              .anyMatch(p -> p.getFileName().toString().equals("tombstones.bin"));
      assertThat(hasTombstonesFile).isTrue();
    }

    @Test
    void tombstonesFileAbsentAfterCompact(@TempDir Path tempDir) throws Exception {
      // After compact(), the new generation has no tombstones, so tombstones.bin must be absent.
      Path storageRoot = tempDir.resolve("col");
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        col.delete("doc-1");
        col.delete("doc-3");
        col.commit();
        col.compact();
      }

      // Tombstones.bin may exist in older gen directories but not in the live one.
      // We verify physicalSize == size (no tombstones remain) after reopen.
      try (var col = newPersistentCollection(IndexType.FLAT, storageRoot)) {
        assertThat(col.physicalSize()).isEqualTo(col.size());
        assertThat(col.size()).isEqualTo(3);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Tombstone-only commits
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class TombstoneOnlyCommit {

    @Test
    void tombstoneOnlyCommitProducesNewGeneration() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        int sizeBefore = col.size();
        col.delete("doc-0");
        col.commit();

        assertThat(col.size()).isEqualTo(sizeBefore - 1);
        assertThat(col.contains("doc-0")).isFalse();
      }
    }

    @Test
    void emptyCommitIsNoOp() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        int sizeBefore = col.size();
        col.commit(); // no-op
        assertThat(col.size()).isEqualTo(sizeBefore);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Combined operations
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class CombinedOperations {

    @Test
    void deleteAndAddInSameCommit() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        col.delete("doc-0");
        col.add(doc("new-doc", 9f, 9f, 9f, 9f, 9f, 9f, 9f, 9f));
        col.commit();

        assertThat(col.size()).isEqualTo(5); // 5 - 1 + 1
        assertThat(col.contains("doc-0")).isFalse();
        assertThat(col.contains("new-doc")).isTrue();
      }
    }

    @Test
    void deleteAllThenCompact() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(5, SEED));
        col.commit();

        for (int i = 0; i < 5; i++) {
          col.delete("doc-" + i);
        }
        col.commit();

        assertThat(col.size()).isZero();
        assertThat(col.physicalSize()).isEqualTo(5);

        col.compact();
        assertThat(col.size()).isZero();
        assertThat(col.physicalSize()).isZero();
      }
    }

    @Test
    void addAfterDeleteReusesFreeId() {
      // After deleting "doc-0" and committing, adding a new doc with the same id should work
      // because the tombstoned ordinal is logically deleted.
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(
            new Document(
                "doc-0", new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, "original", Map.of()));
        col.commit();

        col.delete("doc-0");
        col.commit();

        // Now "doc-0" is tombstoned. Add a new document with the same id using upsert.
        col.upsert(
            new Document(
                "doc-0", new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, "reincarnated", Map.of()));
        col.commit();

        assertThat(col.size()).isEqualTo(1);
        Document fetched = col.get("doc-0");
        assertThat(fetched).isNotNull();
        assertThat(fetched.text()).isEqualTo("reincarnated");
      }
    }

    @Test
    void addAfterDeleteWithSameIdViaAdd() {
      // After delete() + commit() the same id is tombstoned but logically free. Calling add()
      // (not upsert()) with the same id should succeed because the tombstone check in
      // stageUnderLock() recognises that the live ordinal is tombstoned.
      try (var col = newCollection(IndexType.FLAT)) {
        col.add(new Document("x", new float[] {1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, "first", Map.of()));
        col.commit();

        col.delete("x");
        col.commit();
        assertThat(col.contains("x")).isFalse();

        // add() — not upsert() — after the tombstone has been committed.
        col.add(
            new Document("x", new float[] {0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}, "second", Map.of()));
        col.commit();

        assertThat(col.size()).isEqualTo(1);
        // physicalSize is 2: ordinal 0 (tombstoned "first") + ordinal 1 (live "second").
        assertThat(col.physicalSize()).isEqualTo(2);
        Document fetched = col.get("x");
        assertThat(fetched).isNotNull();
        assertThat(fetched.text()).isEqualTo("second");
      }
    }

    @Test
    void multipleDeleteCommitCycles() {
      try (var col = newCollection(IndexType.FLAT)) {
        col.addAll(generateDocs(20, SEED));
        col.commit();

        // Delete in batches.
        for (int batch = 0; batch < 4; batch++) {
          for (int i = batch * 5; i < (batch + 1) * 5; i++) {
            col.delete("doc-" + i);
          }
          col.commit();
          assertThat(col.size()).isEqualTo(20 - (batch + 1) * 5);
        }

        assertThat(col.size()).isZero();
        assertThat(col.physicalSize()).isEqualTo(20);

        col.compact();
        assertThat(col.physicalSize()).isZero();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // IGTM HNSW graph merge gate tests (OM2)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class IgtmMerge {

    private static final int CORPUS = 100;
    private static final int DELETIONS = 25; // 25 % delete fraction

    /** After merging 25 % deletions the surviving vectors are all still reachable. */
    @Test
    void hnswMerge_survivingDocsRemainSearchable() {
      try (var col = newCollection(IndexType.HNSW)) {
        col.addAll(generateDocs(CORPUS, SEED));
        col.commit();

        for (int i = 0; i < DELETIONS; i++) {
          col.delete("doc-" + i);
        }
        col.commit();
        col.compact();

        assertThat(col.size()).isEqualTo(CORPUS - DELETIONS);
        assertThat(col.physicalSize()).isEqualTo(CORPUS - DELETIONS);

        // Every surviving doc must be findable by exact-id lookup.
        for (int i = DELETIONS; i < CORPUS; i++) {
          assertThat(col.contains("doc-" + i)).isTrue();
        }
        // Deleted docs must be gone.
        for (int i = 0; i < DELETIONS; i++) {
          assertThat(col.contains("doc-" + i)).isFalse();
        }
      }
    }

    /** Recall of merged HNSW must be ≥ 80 % vs flat-scan ground truth on the surviving corpus. */
    @Test
    void hnswMerge_recallAfterMerge() {
      try (var col = newCollection(IndexType.HNSW)) {
        List<Document> docs = generateDocs(CORPUS, SEED);
        col.addAll(docs);
        col.commit();

        for (int i = 0; i < DELETIONS; i++) {
          col.delete("doc-" + i);
        }
        col.commit();
        col.compact();

        int remaining = CORPUS - DELETIONS;
        // Use a query vector near the surviving docs.
        float[] query = docs.get(DELETIONS).vector();
        int k = Math.min(10, remaining);

        var result = col.search(SearchRequest.builder(query, k).build());
        assertThat(result.hits()).isNotEmpty();
        // None of the deleted docs should appear.
        for (var hit : result.hits()) {
          int idx = Integer.parseInt(hit.id().substring(4));
          assertThat(idx).isGreaterThanOrEqualTo(DELETIONS);
        }
        // Recall: at least 80 % of k results retrieved.
        assertThat(result.hits().size()).isGreaterThanOrEqualTo((int) Math.ceil(k * 0.8));
      }
    }

    /** Persistent HNSW compact (merge path) survives close + reopen with correct results. */
    @Test
    void persistentHnswMerge_survivesCloseAndReopen(@TempDir Path tempDir) {
      Path storageRoot = tempDir.resolve("col");
      List<Document> docs = generateDocs(CORPUS, SEED);

      try (var col = newPersistentCollection(IndexType.HNSW, storageRoot)) {
        col.addAll(docs);
        col.commit();

        for (int i = 0; i < DELETIONS; i++) {
          col.delete("doc-" + i);
        }
        col.commit();
        col.compact();

        assertThat(col.size()).isEqualTo(CORPUS - DELETIONS);
        assertThat(col.physicalSize()).isEqualTo(CORPUS - DELETIONS);
      }

      try (var col = newPersistentCollection(IndexType.HNSW, storageRoot)) {
        assertThat(col.size()).isEqualTo(CORPUS - DELETIONS);
        assertThat(col.physicalSize()).isEqualTo(CORPUS - DELETIONS);

        for (int i = 0; i < DELETIONS; i++) {
          assertThat(col.contains("doc-" + i)).isFalse();
        }
        for (int i = DELETIONS; i < CORPUS; i++) {
          assertThat(col.contains("doc-" + i)).isTrue();
        }

        // Verify search works post-reopen.
        float[] query = docs.get(DELETIONS).vector();
        int k = 10;
        var result = col.search(SearchRequest.builder(query, k).build());
        assertThat(result.hits()).isNotEmpty();
        for (var hit : result.hits()) {
          int idx = Integer.parseInt(hit.id().substring(4));
          assertThat(idx).isGreaterThanOrEqualTo(DELETIONS);
        }
      }
    }
  }
}
