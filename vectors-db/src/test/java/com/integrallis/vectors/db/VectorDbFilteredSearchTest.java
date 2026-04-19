package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
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

  // ---------------------------------------------------------------------------
  // ACORN pre-filter gate tests (OM3)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class AcornPreFilter {

    private static final int CORPUS = 100;
    private static final int DIM_L = 8;
    private static final long SEED = 42L;

    private List<Document> buildGroupedDocs() {
      Random rng = new Random(SEED);
      List<Document> docs = new ArrayList<>(CORPUS);
      for (int i = 0; i < CORPUS; i++) {
        float[] v = new float[DIM_L];
        for (int d = 0; d < DIM_L; d++) v[d] = (float) rng.nextGaussian();
        String group = i < CORPUS / 2 ? "A" : "B";
        docs.add(new Document("doc-" + i, v, null, Map.of("group", MetadataValue.of(group))));
      }
      return docs;
    }

    private VectorCollection buildHnsw(List<Document> docs) {
      var col =
          VectorCollection.builder()
              .dimension(DIM_L)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build();
      col.addAll(docs);
      col.commit();
      return col;
    }

    /** Pre-filter must never return a group-B doc when filtering for group=A. */
    @Test
    void preFilter_returnsOnlyMatchingDocs() {
      List<Document> docs = buildGroupedDocs();
      try (var col = buildHnsw(docs)) {
        float[] query = new float[DIM_L];
        var result =
            col.search(SearchRequest.builder(query, 10).filter(Filters.eq("group", "A")).build());
        assertThat(result.hits()).isNotEmpty();
        for (var hit : result.hits()) {
          assertThat(Integer.parseInt(hit.id().substring(4))).isLessThan(50);
        }
      }
    }

    /** Recall: top-10 from a 50 %-selective filter must return ≥ 8 results. */
    @Test
    void preFilter_recallIsAdequate() {
      List<Document> docs = buildGroupedDocs();
      try (var col = buildHnsw(docs)) {
        float[] query = docs.get(0).vector();
        var result =
            col.search(SearchRequest.builder(query, 10).filter(Filters.eq("group", "A")).build());
        assertThat(result.hits().size()).isGreaterThanOrEqualTo(8);
        for (var hit : result.hits()) {
          assertThat(Integer.parseInt(hit.id().substring(4))).isLessThan(50);
        }
      }
    }

    /** Persistent HNSW (MappedHnswIndexAdapter) also uses the ACORN path. */
    @Test
    void persistentHnsw_preFilterWorks(@TempDir Path tempDir) {
      List<Document> docs = buildGroupedDocs();
      try (var col =
          VectorCollection.builder()
              .dimension(DIM_L)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .storagePath(tempDir.resolve("acorn"))
              .build()) {
        col.addAll(docs);
        col.commit();
        float[] query = docs.get(0).vector();
        var result =
            col.search(SearchRequest.builder(query, 10).filter(Filters.eq("group", "A")).build());
        assertThat(result.hits().size()).isGreaterThanOrEqualTo(8);
        for (var hit : result.hits()) {
          assertThat(Integer.parseInt(hit.id().substring(4))).isLessThan(50);
        }
      }
    }

    /**
     * Routing property test: with a 10% selective filter, matching nodes are sparse in the graph.
     * Most matching nodes have NO direct edge to another matching node, so the search MUST route
     * through non-matching nodes to discover them. A naive post-filter (unfiltered search then
     * discard) would miss most matches on selective filters.
     *
     * <p>This test uses 500 vectors with only every 10th vector (50 total) in the "target" group.
     * We query for k=5 and verify that the search finds all 5 matching results, proving that
     * non-matching nodes successfully served as routing infrastructure.
     */
    @Test
    void preFilter_routingThroughNonMatchingNodes_highSelectivity() {
      int n = 500;
      int dim = 8;
      Random rng = new Random(SEED);
      List<Document> docs = new ArrayList<>(n);
      for (int i = 0; i < n; i++) {
        float[] v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = (float) rng.nextGaussian();
        // Only every 10th vector is in the "target" group — 10% selectivity
        String group = (i % 10 == 0) ? "target" : "filler";
        docs.add(new Document("doc-" + i, v, null, Map.of("group", MetadataValue.of(group))));
      }

      try (var col =
          VectorCollection.builder()
              .dimension(dim)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build()) {
        col.addAll(docs);
        col.commit();

        // Query with a target vector — should find matching neighbors by routing through fillers
        float[] query = docs.get(0).vector(); // doc-0 is a target
        var result =
            col.search(
                SearchRequest.builder(query, 5).filter(Filters.eq("group", "target")).build());

        // Must find exactly 5 results, all from the target group
        assertThat(result.hits()).hasSize(5);
        for (var hit : result.hits()) {
          int docIdx = Integer.parseInt(hit.id().substring(4));
          assertThat(docIdx % 10)
              .as("Doc %d should be a target (ordinal %% 10 == 0)", docIdx)
              .isEqualTo(0);
        }

        // Verify these are actually the closest target vectors via brute force
        float[] q = query;
        List<Integer> targetIndices = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
          if (i % 10 == 0) targetIndices.add(i);
        }
        targetIndices.sort(
            (a, b) -> {
              float da = SimilarityFunction.EUCLIDEAN.compare(q, docs.get(a).vector());
              float db = SimilarityFunction.EUCLIDEAN.compare(q, docs.get(b).vector());
              return Float.compare(db, da); // descending (higher = better for EUCLIDEAN sim)
            });
        java.util.Set<String> bruteTop5 = new java.util.HashSet<>();
        for (int i = 0; i < 5; i++) bruteTop5.add("doc-" + targetIndices.get(i));

        java.util.Set<String> acornTop5 = new java.util.HashSet<>();
        for (var hit : result.hits()) acornTop5.add(hit.id());

        // ACORN should find at least 4 of the 5 brute-force nearest targets
        int overlap = 0;
        for (String id : acornTop5) if (bruteTop5.contains(id)) overlap++;
        assertThat(overlap)
            .as("ACORN recall@5 among 10%%-selective targets vs brute force")
            .isGreaterThanOrEqualTo(4);
      }
    }
  }
}
