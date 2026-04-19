package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.filter.Filters;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** End-to-end integration tests for the Step 2 vectors-db facade. */
class VectorDbIntegrationTest {

  private static VectorCollection newCollection(int dim, SimilarityFunction metric) {
    return VectorCollection.builder()
        .dimension(dim)
        .metric(metric)
        .indexType(IndexType.FLAT)
        .quantizer(QuantizerKind.NONE)
        .build();
  }

  private static float[][] randomVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] data = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        data[i][d] = rng.nextFloat();
      }
    }
    return data;
  }

  private static List<Document> asDocuments(float[][] vectors) {
    List<Document> docs = new ArrayList<>(vectors.length);
    for (int i = 0; i < vectors.length; i++) {
      docs.add(Document.of("doc-" + i, vectors[i]));
    }
    return docs;
  }

  @Nested
  @Tag("unit")
  class BuilderValidation {

    @Test
    void build_throws_whenDimensionMissing() {
      assertThatIllegalStateException()
          .isThrownBy(
              () ->
                  VectorCollection.builder()
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .indexType(IndexType.FLAT)
                      .build())
          .withMessageContaining("dimension");
    }

    @Test
    void build_throws_whenMetricMissing() {
      assertThatIllegalStateException()
          .isThrownBy(
              () -> VectorCollection.builder().dimension(8).indexType(IndexType.FLAT).build())
          .withMessageContaining("metric");
    }

    @Test
    void build_succeeds_withDimensionAndMetric() {
      try (var col =
          VectorCollection.builder().dimension(8).metric(SimilarityFunction.EUCLIDEAN).build()) {
        assertThat(col.size()).isZero();
        assertThat(col.config().dimension()).isEqualTo(8);
        assertThat(col.config().metric()).isEqualTo(SimilarityFunction.EUCLIDEAN);
        assertThat(col.config().indexType()).isEqualTo(IndexType.FLAT);
        assertThat(col.config().quantizerKind()).isEqualTo(QuantizerKind.NONE);
      }
    }

    @Test
    void indexTypeHnsw_buildsSuccessfully_inStep4b() {
      // Step 4b unblocked HNSW in both in-memory and persistent modes. The 5-arg builder here
      // runs the in-memory path; persistent HNSW is covered by VectorDbHnswPersistenceTest.
      try (var col =
          VectorCollection.builder()
              .dimension(8)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.HNSW)
              .build()) {
        assertThat(col.size()).isZero();
        assertThat(col.config().indexType()).isEqualTo(IndexType.HNSW);
        assertThat(col.config().hnswParams()).isNotNull();
      }
    }

    @Test
    void indexTypeVamana_buildsSuccessfully_inStep4c() {
      // Step 4c unblocked VAMANA in both in-memory and persistent modes. The 5-arg builder here
      // runs the in-memory path; persistent VAMANA is covered by VectorDbVamanaPersistenceTest.
      try (var col =
          VectorCollection.builder()
              .dimension(8)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.VAMANA)
              .build()) {
        assertThat(col.size()).isZero();
        assertThat(col.config().indexType()).isEqualTo(IndexType.VAMANA);
        assertThat(col.config().vamanaParams()).isNotNull();
      }
    }

    @Test
    void quantizerKindSq8_throwsUnsupported_inStep2() {
      assertThatExceptionOfType(UnsupportedOperationException.class)
          .isThrownBy(
              () ->
                  VectorCollection.builder()
                      .dimension(8)
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .quantizer(QuantizerKind.SQ8)
                      .build())
          .withMessageContaining("SQ8");
    }
  }

  @Nested
  @Tag("unit")
  class Lifecycle {

    @Test
    void addBeforeCommit_searchSeesZeroResults() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        var result = col.search(SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 5).build());
        assertThat(result.hits()).isEmpty();
      }
    }

    @Test
    void addAndCommit_searchSeesAllDocs() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 0f, 0f, 0f}));
        col.add(Document.of("b", new float[] {0f, 1f, 0f, 0f}));
        col.add(Document.of("c", new float[] {0f, 0f, 1f, 0f}));
        col.commit();

        assertThat(col.size()).isEqualTo(3);
        var result = col.search(SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 3).build());
        assertThat(result.hits()).hasSize(3);
        assertThat(result.hits().get(0).id()).isEqualTo("a");
      }
    }

    @Test
    void addAll_equivalentToRepeatedAdd() {
      try (var a = newCollection(4, SimilarityFunction.EUCLIDEAN);
          var b = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        List<Document> docs =
            List.of(
                Document.of("a", new float[] {1f, 0f, 0f, 0f}),
                Document.of("b", new float[] {0f, 1f, 0f, 0f}),
                Document.of("c", new float[] {0f, 0f, 1f, 0f}));

        for (Document d : docs) a.add(d);
        a.commit();

        b.addAll(docs);
        b.commit();

        var req = SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 3).build();
        var ra = a.search(req);
        var rb = b.search(req);
        assertThat(rb.hits()).hasSameSizeAs(ra.hits());
        for (int i = 0; i < ra.hits().size(); i++) {
          assertThat(rb.hits().get(i).id()).isEqualTo(ra.hits().get(i).id());
          assertThat(rb.hits().get(i).score()).isEqualTo(ra.hits().get(i).score());
        }
      }
    }

    @Test
    void close_releasesResources() {
      var col = newCollection(4, SimilarityFunction.EUCLIDEAN);
      col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
      col.commit();
      col.close();

      assertThatIllegalStateException()
          .isThrownBy(
              () -> col.search(SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1).build()))
          .withMessageContaining("closed");
    }

    @Test
    void addDuplicateId_throws() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        assertThatIllegalArgumentException()
            .isThrownBy(() -> col.add(Document.of("a", new float[] {5f, 6f, 7f, 8f})))
            .withMessageContaining("Duplicate id");
      }
    }

    @Test
    void addWrongDimension_throws() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        assertThatIllegalArgumentException()
            .isThrownBy(() -> col.add(Document.of("a", new float[] {1f, 2f, 3f})))
            .withMessageContaining("dimension");
      }
    }
  }

  @Nested
  @Tag("unit")
  class DocumentRoundTrip {

    @Test
    void allMetadataValueTypes_preservedThroughSearch() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        Map<String, MetadataValue> metadata =
            Map.of(
                "title", MetadataValue.of("hello"),
                "count", MetadataValue.of(42L),
                "ratio", MetadataValue.of(3.14d),
                "active", MetadataValue.of(true),
                "tags", MetadataValue.tags("a", "b", "c"));
        var doc = new Document("doc-1", new float[] {1f, 2f, 3f, 4f}, "raw text", metadata);
        col.add(doc);
        col.commit();

        var result = col.search(SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1).build());
        assertThat(result.hits()).hasSize(1);

        var hit = result.hits().get(0);
        assertThat(hit.id()).isEqualTo("doc-1");
        assertThat(hit.document().text()).isEqualTo("raw text");

        Map<String, MetadataValue> returned = hit.document().metadata();
        assertThat(returned.get("title")).isInstanceOf(MetadataValue.Str.class);
        assertThat(((MetadataValue.Str) returned.get("title")).value()).isEqualTo("hello");

        assertThat(returned.get("count")).isInstanceOf(MetadataValue.Num.class);
        assertThat(((MetadataValue.Num) returned.get("count")).value()).isEqualTo(42d);

        assertThat(returned.get("ratio")).isInstanceOf(MetadataValue.Num.class);
        assertThat(((MetadataValue.Num) returned.get("ratio")).value()).isEqualTo(3.14d);

        assertThat(returned.get("active")).isInstanceOf(MetadataValue.Bool.class);
        assertThat(((MetadataValue.Bool) returned.get("active")).value()).isTrue();

        assertThat(returned.get("tags")).isInstanceOf(MetadataValue.Tags.class);
        assertThat(((MetadataValue.Tags) returned.get("tags")).values())
            .containsExactly("a", "b", "c");
      }
    }

    /**
     * Regression for Step 4a audit finding C3 — {@code commitInMemory} used to share the caller's
     * {@code float[]} with the FlatScanAdapter, which meant a later mutation to the array (held
     * through the original Document) would silently corrupt the index. The fix defensively clones
     * each vector into the successor matrix on commit so the index is insulated from external
     * mutation.
     */
    @Test
    void mutatingOriginalFloatArrayAfterCommitDoesNotCorruptIndex() {
      float[] userOwnedVector = new float[] {1f, 2f, 3f, 4f};
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(new Document("id", userOwnedVector, null, Map.of()));
        col.commit();

        // Mutate the user-owned array after the commit. Prior to the C3 fix this would corrupt
        // the FlatScanAdapter's vector matrix and the search would return a different score.
        userOwnedVector[0] = 999f;
        userOwnedVector[1] = 999f;
        userOwnedVector[2] = 999f;
        userOwnedVector[3] = 999f;

        var result = col.search(SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1).build());
        assertThat(result.hits()).hasSize(1);
        // EUCLIDEAN similarity = 1 / (1 + L2^2). For an exact match L2^2=0 → score=1.0. If the
        // adapter saw the mutated vector instead, the distance would be enormous and the score
        // would be far from 1.0.
        assertThat(result.hits().get(0).score()).isEqualTo(1f);
      }
    }
  }

  @Nested
  @Tag("unit")
  class FlatScanSearch {

    @ParameterizedTest
    @EnumSource(SimilarityFunction.class)
    void top10_matchesBruteForceGroundTruth_exactly(SimilarityFunction sim) {
      float[][] data = randomVectors(1000, 16, 42L);
      try (var col = newCollection(16, sim)) {
        col.addAll(asDocuments(data));
        col.commit();

        float[] query = data[7];
        var result = col.search(SearchRequest.builder(query, 10).build());
        assertThat(result.hits()).hasSize(10);

        // Brute-force: flat scan IS brute force, so the results must match bit-for-bit.
        int[] expected = bruteForceTopK(data, query, 10, sim);
        for (int i = 0; i < 10; i++) {
          assertThat(result.hits().get(i).id()).isEqualTo("doc-" + expected[i]);
          // And the score must equal metric.compare exactly.
          assertThat(result.hits().get(i).score()).isEqualTo(sim.compare(query, data[expected[i]]));
        }
      }
    }

    private int[] bruteForceTopK(float[][] data, float[] query, int k, SimilarityFunction sim) {
      Integer[] order = new Integer[data.length];
      for (int i = 0; i < data.length; i++) order[i] = i;
      float[] scores = new float[data.length];
      for (int i = 0; i < data.length; i++) scores[i] = sim.compare(query, data[i]);
      java.util.Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
      int[] out = new int[k];
      for (int i = 0; i < k; i++) out[i] = order[i];
      return out;
    }
  }

  @Nested
  @Tag("unit")
  class SearchRequestDefaults {

    @Test
    void builderAppliesDocumentedDefaults() {
      float[] q = new float[] {1f, 2f, 3f, 4f};
      var req = SearchRequest.builder(q, 7).build();
      assertThat(req.k()).isEqualTo(7);
      assertThat(req.searchListSize()).isEqualTo(Math.max(7, 100));
      assertThat(req.overQueryFactor()).isEqualTo(4.0f);
      assertThat(req.filterExpansion()).isEqualTo(4.0f);
      assertThat(req.minScore()).isEqualTo(-Float.MAX_VALUE);
      assertThat(req.filter()).isNull();
      assertThat(req.includeVector()).isTrue();
      assertThat(req.includeText()).isTrue();
      assertThat(req.includeMetadata()).isTrue();
    }

    @Test
    void includeVectorFalse_returnsNullVectorOnHits() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        col.commit();

        var req =
            SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1).includeVector(false).build();
        var result = col.search(req);
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().get(0).document().vector()).isNull();
      }
    }

    @Test
    void minScore_filtersResults() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("near", new float[] {1f, 2f, 3f, 4f}));
        col.add(Document.of("far", new float[] {100f, 200f, 300f, 400f}));
        col.commit();

        // Perfect self-match scores 1.0; the distant vector scores near zero.
        // minScore=0.5 should exclude the distant vector.
        var req = SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 2).minScore(0.5f).build();
        var result = col.search(req);
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().get(0).id()).isEqualTo("near");
      }
    }
  }

  @Nested
  @Tag("unit")
  class DeterministicResults {

    @Test
    void sameDataSameQuery_producesIdenticalResults() {
      float[][] data = randomVectors(200, 16, 42L);

      try (var a = newCollection(16, SimilarityFunction.EUCLIDEAN);
          var b = newCollection(16, SimilarityFunction.EUCLIDEAN)) {
        a.addAll(asDocuments(data));
        a.commit();
        b.addAll(asDocuments(data));
        b.commit();

        var req = SearchRequest.builder(data[11], 10).build();
        var ra = a.search(req);
        var rb = b.search(req);
        assertThat(rb.hits()).hasSameSizeAs(ra.hits());
        for (int i = 0; i < ra.hits().size(); i++) {
          assertThat(rb.hits().get(i).id()).isEqualTo(ra.hits().get(i).id());
          assertThat(rb.hits().get(i).score()).isEqualTo(ra.hits().get(i).score());
        }
      }
    }
  }

  @Nested
  @Tag("unit")
  class AutoCommit {

    @Test
    void defaultConfigNeverAutoCommits() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        // Default builder leaves autoCommitThreshold at Integer.MAX_VALUE → no auto-commit.
        List<Document> docs = new ArrayList<>(1000);
        for (int i = 0; i < 1000; i++) {
          docs.add(Document.of("doc-" + i, new float[] {i, 0f, 0f, 0f}));
        }
        col.addAll(docs);

        // Nothing should be visible to search() yet.
        assertThat(col.size()).isZero();
        var result = col.search(SearchRequest.builder(new float[] {0f, 0f, 0f, 0f}, 10).build());
        assertThat(result.hits()).isEmpty();

        // Explicit commit makes them visible.
        col.commit();
        assertThat(col.size()).isEqualTo(1000);
      }
    }

    @Test
    void addAllBeyondThresholdAutoCommits() {
      try (var col =
          VectorCollection.builder()
              .dimension(4)
              .metric(SimilarityFunction.EUCLIDEAN)
              .autoCommitThreshold(100)
              .build()) {
        List<Document> docs = new ArrayList<>(200);
        for (int i = 0; i < 200; i++) {
          docs.add(Document.of("doc-" + i, new float[] {i, 0f, 0f, 0f}));
        }
        // addAll batches all 200 and then runs a single auto-commit on the way out.
        col.addAll(docs);

        // NO explicit commit() call.
        assertThat(col.size()).isEqualTo(200);
        var result = col.search(SearchRequest.builder(new float[] {0f, 0f, 0f, 0f}, 200).build());
        assertThat(result.hits()).hasSize(200);
      }
    }

    @Test
    void individualAddBeyondThresholdAutoCommits() {
      try (var col =
          VectorCollection.builder()
              .dimension(4)
              .metric(SimilarityFunction.EUCLIDEAN)
              .autoCommitThreshold(5)
              .build()) {
        for (int i = 0; i < 10; i++) {
          col.add(Document.of("doc-" + i, new float[] {i, 0f, 0f, 0f}));
        }
        // No explicit commit — auto-commit fired at size=5 and then again at size=5 after reset.
        assertThat(col.size()).isEqualTo(10);
        var result = col.search(SearchRequest.builder(new float[] {0f, 0f, 0f, 0f}, 10).build());
        assertThat(result.hits()).hasSize(10);
      }
    }
  }

  @Nested
  @Tag("unit")
  class FilterExecution {

    @Test
    void nullFilter_succeeds() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        col.commit();

        var req = SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1).filter(null).build();
        var result = col.search(req);
        assertThat(result.hits()).hasSize(1);
      }
    }

    @Test
    void filterAll_succeeds_withIdenticalResultsToNull() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        col.add(Document.of("b", new float[] {2f, 3f, 4f, 5f}));
        col.commit();

        var nullFilter =
            col.search(SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 2).filter(null).build());
        var allFilter =
            col.search(
                SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 2)
                    .filter(Filters.all())
                    .build());

        assertThat(allFilter.hits()).hasSameSizeAs(nullFilter.hits());
        for (int i = 0; i < nullFilter.hits().size(); i++) {
          assertThat(allFilter.hits().get(i).id()).isEqualTo(nullFilter.hits().get(i).id());
          assertThat(allFilter.hits().get(i).score()).isEqualTo(nullFilter.hits().get(i).score());
        }
      }
    }

    @Test
    void nonTrivialFilter_returnsMatchingDocumentsOnly() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(
            new Document(
                "a", new float[] {1f, 2f, 3f, 4f}, null, Map.of("color", MetadataValue.of("red"))));
        col.add(
            new Document(
                "b",
                new float[] {1f, 2f, 3f, 5f},
                null,
                Map.of("color", MetadataValue.of("blue"))));
        col.commit();

        var req =
            SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 10)
                .filter(Filters.eq("color", "red"))
                .build();
        var result = col.search(req);
        assertThat(result.hits()).hasSize(1);
        assertThat(result.hits().getFirst().id()).isEqualTo("a");
      }
    }

    @Test
    void filterOnMissingField_returnsEmpty() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        col.commit();

        var req =
            SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1)
                .filter(Filters.eq("nonexistent", "value"))
                .build();
        var result = col.search(req);
        assertThat(result.hits()).isEmpty();
      }
    }
  }

  @Nested
  @Tag("unit")
  class BatchSearch {

    @Test
    void searchBatch_nullRequests_throws() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        assertThatIllegalArgumentException().isThrownBy(() -> col.searchBatch(null));
      }
    }

    @Test
    void searchBatch_emptyRequests_throws() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        assertThatIllegalArgumentException().isThrownBy(() -> col.searchBatch(List.of()));
      }
    }

    @Test
    void searchBatch_singleRequest_matchesSingleSearch() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("x", new float[] {1f, 0f, 0f, 0f}));
        col.commit();
        var req = SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 1).build();
        var single = col.search(req);
        var batch = col.searchBatch(List.of(req));
        assertThat(batch).hasSize(1);
        assertThat(batch.get(0).hits().get(0).id()).isEqualTo(single.hits().get(0).id());
      }
    }

    @Test
    void searchBatch_multipleRequests_resultsInOrder() {
      int dim = 32;
      float[][] data = randomVectors(100, dim, 1L);
      try (var col =
          VectorCollection.builder()
              .dimension(dim)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.FLAT)
              .build()) {
        col.addAll(asDocuments(data));
        col.commit();

        // Build 8 independent queries; verify each batch result matches individual search
        List<SearchRequest> requests = new ArrayList<>();
        List<SearchResult> expected = new ArrayList<>();
        float[][] queries = randomVectors(8, dim, 99L);
        for (float[] q : queries) {
          var req = SearchRequest.builder(q, 5).build();
          requests.add(req);
          expected.add(col.search(req));
        }

        List<SearchResult> batch = col.searchBatch(requests);
        assertThat(batch).hasSize(requests.size());
        for (int i = 0; i < requests.size(); i++) {
          var batchHitIds = batch.get(i).hits().stream().map(SearchResult.Hit::id).toList();
          var expectedHitIds = expected.get(i).hits().stream().map(SearchResult.Hit::id).toList();
          assertThat(batchHitIds)
              .as("batch result[%d] should match individual search", i)
              .containsExactlyElementsOf(expectedHitIds);
        }
      }
    }

    @Test
    void searchBatch_resultsListIsImmutable() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 0f, 0f, 0f}));
        col.commit();
        var req = SearchRequest.builder(new float[] {1f, 0f, 0f, 0f}, 1).build();
        var results = col.searchBatch(List.of(req));
        assertThatExceptionOfType(UnsupportedOperationException.class)
            .isThrownBy(() -> results.add(null));
      }
    }

    @Test
    void searchBatch_concurrencyStressTest_noResultCorruption() {
      int dim = 32;
      float[][] data = randomVectors(200, dim, 7L);
      try (var col =
          VectorCollection.builder()
              .dimension(dim)
              .metric(SimilarityFunction.EUCLIDEAN)
              .indexType(IndexType.FLAT)
              .build()) {
        col.addAll(asDocuments(data));
        col.commit();

        // Fire 32 queries as a single batch; all should return k=10 hits
        List<SearchRequest> requests = new ArrayList<>();
        for (float[] q : randomVectors(32, dim, 77L)) {
          requests.add(SearchRequest.builder(q, 10).build());
        }
        var results = col.searchBatch(requests);
        assertThat(results).hasSize(32);
        for (var r : results) {
          assertThat(r.hits()).hasSize(10);
        }
      }
    }
  }
}
