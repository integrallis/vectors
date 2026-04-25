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
package com.integrallis.vectors.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;

class JavaVectorsVectorStoreTest {

  static final int DIMENSION = 4;

  EmbeddingModel embeddingModel;
  VectorCollection collection;
  JavaVectorsVectorStore store;

  /** Deterministic embedding: hash the text to produce a unit-ish vector. */
  static float[] deterministicEmbedding(String text) {
    int hash = text.hashCode();
    float[] vec = new float[DIMENSION];
    for (int i = 0; i < DIMENSION; i++) {
      vec[i] = ((hash >> (i * 8)) & 0xFF) / 255.0f;
    }
    // Normalize
    float norm = 0;
    for (float v : vec) norm += v * v;
    norm = (float) Math.sqrt(norm);
    if (norm > 0) {
      for (int i = 0; i < DIMENSION; i++) vec[i] /= norm;
    }
    return vec;
  }

  @BeforeEach
  void setUp() {
    embeddingModel = mock(EmbeddingModel.class);
    when(embeddingModel.dimensions()).thenReturn(DIMENSION);
    when(embeddingModel.embed(any(Document.class)))
        .thenAnswer(
            inv -> {
              Document doc = inv.getArgument(0);
              String text = doc.getText() != null ? doc.getText() : doc.getId();
              return deterministicEmbedding(text);
            });
    when(embeddingModel.embed(any(String.class)))
        .thenAnswer(inv -> deterministicEmbedding(inv.getArgument(0)));
  }

  @AfterEach
  void tearDown() {
    if (store != null) {
      store.close();
    }
    if (collection != null) {
      collection.close();
    }
  }

  private void createStore(IndexType indexType, boolean commitAfterAdd) {
    collection =
        VectorCollection.builder()
            .dimension(DIMENSION)
            .metric(SimilarityFunction.COSINE)
            .indexType(indexType)
            .autoCommitThreshold(Integer.MAX_VALUE)
            .build();
    store =
        JavaVectorsVectorStore.builder(embeddingModel, collection)
            .commitAfterAdd(commitAfterAdd)
            .build();
  }

  @Nested
  @Tag("unit")
  class AddAndSearch {

    @Test
    void addAndSearchRoundTrip() {
      createStore(IndexType.FLAT, true);

      Document doc = new Document("doc-1", "The quick brown fox", Map.of("animal", "fox"));
      store.add(List.of(doc));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("quick brown fox").topK(1).build());

      assertThat(results).hasSize(1);
      assertThat(results.getFirst().getId()).isEqualTo("doc-1");
    }

    @Test
    void multipleDocuments() {
      createStore(IndexType.FLAT, true);

      store.add(
          List.of(
              new Document("d1", "alpha beta", Map.of()),
              new Document("d2", "gamma delta", Map.of()),
              new Document("d3", "epsilon zeta", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("alpha beta").topK(3).build());

      assertThat(results).hasSize(3);
      // All results should have positive scores
      results.forEach(r -> assertThat(r.getScore()).isGreaterThan(0.0));
    }

    @Test
    void topKLimitsResults() {
      createStore(IndexType.FLAT, true);

      store.add(
          List.of(
              new Document("d1", "one", Map.of()),
              new Document("d2", "two", Map.of()),
              new Document("d3", "three", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("one").topK(2).build());

      assertThat(results).hasSize(2);
    }

    @Test
    void similarityThresholdFilters() {
      createStore(IndexType.FLAT, true);

      store.add(
          List.of(
              new Document("d1", "very relevant query text", Map.of()),
              new Document("d2", "completely unrelated content xyz", Map.of())));

      // Very high threshold should return fewer results
      List<Document> highThreshold =
          store.similaritySearch(
              SearchRequest.builder()
                  .query("very relevant query text")
                  .topK(10)
                  .similarityThreshold(0.99)
                  .build());

      List<Document> lowThreshold =
          store.similaritySearch(
              SearchRequest.builder()
                  .query("very relevant query text")
                  .topK(10)
                  .similarityThreshold(0.0)
                  .build());

      assertThat(highThreshold.size()).isLessThanOrEqualTo(lowThreshold.size());
    }

    @Test
    void emptyAddIsNoOp() {
      createStore(IndexType.FLAT, true);
      store.add(List.of());
      assertThat(collection.size()).isZero();
    }
  }

  @Nested
  @Tag("unit")
  class MetadataPreservation {

    @Test
    void metadataRoundTrip() {
      createStore(IndexType.FLAT, true);

      Document doc =
          new Document(
              "md-1",
              "metadata test",
              Map.of("str_key", "str_val", "num_key", 42, "bool_key", true));
      store.add(List.of(doc));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("metadata test").topK(1).build());

      assertThat(results).hasSize(1);
      Map<String, Object> md = results.getFirst().getMetadata();
      assertThat(md.get("str_key")).isEqualTo("str_val");
      assertThat(md.get("num_key")).isEqualTo(42L);
      assertThat(md.get("bool_key")).isEqualTo(true);
    }
  }

  @Nested
  @Tag("unit")
  class DeleteBehavior {

    @Test
    void deleteRemovesDocFromSearch() {
      createStore(IndexType.FLAT, true);

      store.add(
          List.of(
              new Document("d1", "alpha text", Map.of()),
              new Document("d2", "beta text", Map.of())));

      // Both docs should be searchable.
      List<Document> before =
          store.similaritySearch(SearchRequest.builder().query("alpha text").topK(10).build());
      assertThat(before).hasSize(2);

      // Delete one doc.
      store.delete(List.of("d1"));

      // Only d2 should remain.
      List<Document> after =
          store.similaritySearch(SearchRequest.builder().query("alpha text").topK(10).build());
      assertThat(after).hasSize(1);
      assertThat(after.getFirst().getId()).isEqualTo("d2");
    }

    @Test
    void deleteOfUnknownIdIsNoOp() {
      createStore(IndexType.FLAT, true);

      store.add(List.of(new Document("d1", "some text", Map.of())));

      // Delete a non-existent id — should not throw.
      store.delete(List.of("nonexistent"));

      // Original doc should still be searchable.
      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("some text").topK(1).build());
      assertThat(results).hasSize(1);
    }
  }

  @Nested
  @Tag("unit")
  class CommitStrategy {

    @Test
    void commitAfterAddTrueGivesImmediateSearchability() {
      createStore(IndexType.FLAT, true);

      store.add(List.of(new Document("c1", "commit test", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("commit test").topK(1).build());

      assertThat(results).isNotEmpty();
    }

    @Test
    void commitAfterAddFalseRequiresExplicitCommit() {
      createStore(IndexType.FLAT, false);

      store.add(List.of(new Document("c2", "deferred commit", Map.of())));

      // Not committed yet -- search should return empty
      List<Document> beforeCommit =
          store.similaritySearch(SearchRequest.builder().query("deferred commit").topK(1).build());
      assertThat(beforeCommit).isEmpty();

      // Now commit explicitly
      collection.commit();

      List<Document> afterCommit =
          store.similaritySearch(SearchRequest.builder().query("deferred commit").topK(1).build());
      assertThat(afterCommit).isNotEmpty();
    }
  }

  @Nested
  @Tag("unit")
  class IndexTypes {

    @Test
    void flatIndex() {
      createStore(IndexType.FLAT, true);
      store.add(List.of(new Document("f1", "flat test", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("flat test").topK(1).build());
      assertThat(results).hasSize(1);
    }

    @Test
    void hnswIndex() {
      createStore(IndexType.HNSW, true);

      // Need enough docs for HNSW to build a graph
      store.add(
          List.of(
              new Document("h1", "hnsw alpha", Map.of()),
              new Document("h2", "hnsw beta", Map.of()),
              new Document("h3", "hnsw gamma", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("hnsw alpha").topK(2).build());
      assertThat(results).isNotEmpty();
      assertThat(results.size()).isLessThanOrEqualTo(2);
    }

    @Test
    void vamanaIndex() {
      createStore(IndexType.VAMANA, true);

      store.add(
          List.of(
              new Document("v1", "vamana alpha", Map.of()),
              new Document("v2", "vamana beta", Map.of()),
              new Document("v3", "vamana gamma", Map.of())));

      List<Document> results =
          store.similaritySearch(SearchRequest.builder().query("vamana alpha").topK(2).build());
      assertThat(results).isNotEmpty();
      assertThat(results.size()).isLessThanOrEqualTo(2);
    }
  }

  @Nested
  @Tag("unit")
  class BuilderValidation {

    @Test
    void dimensionMismatchThrows() {
      VectorCollection mismatchedCollection =
          VectorCollection.builder()
              .dimension(8) // Mismatches embeddingModel.dimensions() which returns 4
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .build();

      assertThatThrownBy(
              () -> JavaVectorsVectorStore.builder(embeddingModel, mismatchedCollection).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("4")
          .hasMessageContaining("8");

      mismatchedCollection.close();
    }

    @Test
    void matchingDimensionsSucceeds() {
      createStore(IndexType.FLAT, true);
      // No exception — store was created successfully
      assertThat(store).isNotNull();
    }
  }

  @Nested
  @Tag("unit")
  class CloseAndLifecycle {

    @Test
    void closeIsIdempotent() {
      createStore(IndexType.FLAT, true);
      store.close();
      store.close(); // should not throw
      store = null; // prevent double close in tearDown
    }
  }
}
