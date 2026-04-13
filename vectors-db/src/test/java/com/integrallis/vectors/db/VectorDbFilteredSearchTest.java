package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for post-filter execution in {@link VectorCollection#search}. Exercises all filter types
 * via the public API across FLAT, HNSW, and VAMANA index types, plus persistent (mmap) mode.
 */
class VectorDbFilteredSearchTest {

  private static final int DIM = 4;

  private static VectorCollection newCollection(IndexType indexType) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.COSINE)
        .indexType(indexType)
        .build();
  }

  /** Adds a corpus of documents with varied metadata for filter testing. */
  private static void addCorpus(VectorCollection col) {
    col.add(
        new Document(
            "d1",
            new float[] {1f, 0f, 0f, 0f},
            "alpha",
            Map.of(
                "color", MetadataValue.of("red"),
                "size", MetadataValue.of(10.0),
                "active", MetadataValue.of(true),
                "tags", MetadataValue.tags("a", "b"))));
    col.add(
        new Document(
            "d2",
            new float[] {0f, 1f, 0f, 0f},
            "beta",
            Map.of(
                "color", MetadataValue.of("blue"),
                "size", MetadataValue.of(20.0),
                "active", MetadataValue.of(false),
                "tags", MetadataValue.tags("b", "c"))));
    col.add(
        new Document(
            "d3",
            new float[] {0f, 0f, 1f, 0f},
            "gamma",
            Map.of(
                "color", MetadataValue.of("red"),
                "size", MetadataValue.of(30.0),
                "active", MetadataValue.of(true),
                "tags", MetadataValue.tags("c", "d"))));
    col.add(
        new Document(
            "d4",
            new float[] {0f, 0f, 0f, 1f},
            "delta",
            Map.of(
                "color", MetadataValue.of("green"),
                "size", MetadataValue.of(40.0),
                "active", MetadataValue.of(false))));
    col.commit();
  }

  @Nested
  @Tag("unit")
  class FlatFiltered {

    @Test
    void eqStringFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "red"))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void eqBooleanFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("active", true))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void numericRangeFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.between("size", 15.0, 35.0))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d2", "d3");
      }
    }

    @Test
    void gtFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.gt("size", 30.0))
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d4");
      }
    }

    @Test
    void lteFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.lte("size", 20.0))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d2");
      }
    }

    @Test
    void inStringFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.inStr("color", "red", "green"))
                    .build());
        assertThat(result.hits()).hasSize(3);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3", "d4");
      }
    }

    @Test
    void inNumericFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.inNum("size", 10.0, 40.0))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d4");
      }
    }

    @Test
    void andFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.and(Filters.eq("color", "red"), Filters.gt("size", 15.0)))
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d3");
      }
    }

    @Test
    void orFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.or(Filters.eq("color", "green"), Filters.eq("color", "blue")))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d2", "d4");
      }
    }

    @Test
    void notFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.not(Filters.eq("color", "red")))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d2", "d4");
      }
    }

    @Test
    void nestedCompoundFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        // (color == "red" AND size > 15) OR color == "green"
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(
                        Filters.or(
                            Filters.and(Filters.eq("color", "red"), Filters.gt("size", 15.0)),
                            Filters.eq("color", "green")))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d3", "d4");
      }
    }

    @Test
    void tagsEqFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("tags", "c"))
                    .build());
        // d2 has tags [b,c], d3 has tags [c,d]
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d2", "d3");
      }
    }

    @Test
    void tagsInFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.inStr("tags", "a", "d"))
                    .build());
        // d1 has tags [a,b], d3 has tags [c,d]
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void notInFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        // NOT IN ("red", "blue") → only d4 (green)
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.not(Filters.inStr("color", "red", "blue")))
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d4");
      }
    }

    @Test
    void highSelectivityFilter_returnsFewerThanK() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "green"))
                    .build());
        // Only d4 matches
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d4");
      }
    }

    @Test
    void filterPlusMinScore() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        // Only red documents (d1, d3), and also require high minScore.
        // COSINE.compare returns (1 + cosine) / 2: d1 vs query → 1.0, d3 vs query → 0.5.
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "red"))
                    .minScore(0.9f)
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d1");
      }
    }

    @Test
    void filterPlusTopK() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 1)
                    .filter(Filters.eq("color", "red"))
                    .build());
        // 2 red docs, but topK=1
        assertThat(result.hits()).hasSize(1);
      }
    }

    @Test
    void noDocumentsMatchFilter() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "purple"))
                    .build());
        assertThat(result.hits()).isEmpty();
      }
    }

    @Test
    void filterOnDocumentsWithoutThatField_returnsOnlyMatching() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        // d4 does not have "tags" field
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("tags", "a"))
                    .build());
        // Only d1 has tag "a"
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d1");
      }
    }

    @Test
    void allFilterSameAsNull() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var query = new float[] {1f, 0f, 0f, 0f};
        var noFilter = col.search(SearchRequest.builder(query, 10).build());
        var allFilter = col.search(SearchRequest.builder(query, 10).filter(Filters.all()).build());
        assertThat(allFilter.hits()).hasSameSizeAs(noFilter.hits());
      }
    }

    @Test
    void metadataPreservedInFilteredResults() {
      try (var col = newCollection(IndexType.FLAT)) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 1)
                    .filter(Filters.eq("color", "red"))
                    .build());
        assertThat(result.hits()).hasSize(1);
        Document doc = result.hits().getFirst().document();
        assertThat(doc.metadata().get("color")).isEqualTo(MetadataValue.of("red"));
      }
    }
  }

  @Nested
  @Tag("unit")
  class HnswFiltered {

    private VectorCollection newHnswCollection() {
      return VectorCollection.builder()
          .dimension(DIM)
          .metric(SimilarityFunction.COSINE)
          .indexType(IndexType.HNSW)
          .build();
    }

    @Test
    void eqStringFilter() {
      try (var col = newHnswCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "red"))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void andFilter() {
      try (var col = newHnswCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.and(Filters.eq("active", true), Filters.lt("size", 25.0)))
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d1");
      }
    }

    @Test
    void noMatchReturnsEmpty() {
      try (var col = newHnswCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "purple"))
                    .build());
        assertThat(result.hits()).isEmpty();
      }
    }
  }

  @Nested
  @Tag("unit")
  class VamanaFiltered {

    private VectorCollection newVamanaCollection() {
      return VectorCollection.builder()
          .dimension(DIM)
          .metric(SimilarityFunction.COSINE)
          .indexType(IndexType.VAMANA)
          .build();
    }

    @Test
    void eqStringFilter() {
      try (var col = newVamanaCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "red"))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void numericRangeFilter() {
      try (var col = newVamanaCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.gte("size", 20.0))
                    .build());
        assertThat(result.hits()).hasSize(3);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d2", "d3", "d4");
      }
    }

    @Test
    void noMatchReturnsEmpty() {
      try (var col = newVamanaCollection()) {
        addCorpus(col);
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "purple"))
                    .build());
        assertThat(result.hits()).isEmpty();
      }
    }
  }

  @Nested
  @Tag("unit")
  class PersistentFiltered {

    @Test
    void filteredSearchAfterReopenFlat(@TempDir Path dir) {
      // Write, close, reopen, then search with filter.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .storagePath(dir)
              .build()) {
        addCorpus(col);
      }
      // Reopen from disk.
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .storagePath(dir)
              .build()) {
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.eq("color", "red"))
                    .build());
        assertThat(result.hits()).hasSize(2);
        assertThat(result.hits()).extracting(SearchResult.Hit::id).containsOnly("d1", "d3");
      }
    }

    @Test
    void filteredSearchAfterReopenHnsw(@TempDir Path dir) {
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.HNSW)
              .storagePath(dir)
              .build()) {
        addCorpus(col);
      }
      try (var col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.HNSW)
              .storagePath(dir)
              .build()) {
        var result =
            col.search(
                SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 10)
                    .filter(Filters.and(Filters.eq("color", "red"), Filters.gte("size", 20.0)))
                    .build());
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("d3");
      }
    }
  }
}
