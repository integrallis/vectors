package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
    void indexTypeHnsw_throwsUnsupported_inStep2() {
      assertThatExceptionOfType(UnsupportedOperationException.class)
          .isThrownBy(
              () ->
                  VectorCollection.builder()
                      .dimension(8)
                      .metric(SimilarityFunction.EUCLIDEAN)
                      .indexType(IndexType.HNSW)
                      .build())
          .withMessageContaining("HNSW");
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
  class FilterStubs {

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
    void nonTrivialFilter_throwsUnsupported() {
      try (var col = newCollection(4, SimilarityFunction.EUCLIDEAN)) {
        col.add(Document.of("a", new float[] {1f, 2f, 3f, 4f}));
        col.commit();

        var req =
            SearchRequest.builder(new float[] {1f, 2f, 3f, 4f}, 1)
                .filter(Filters.eq("foo", "bar"))
                .build();
        assertThatThrownBy(() -> col.search(req))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Step 5");
      }
    }
  }
}
