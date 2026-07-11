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
package com.integrallis.vectors.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class JavaVectorsEmbeddingStoreTest {

  static final int DIMENSION = 4;

  VectorCollection collection;
  JavaVectorsEmbeddingStore store;

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
    store = JavaVectorsEmbeddingStore.builder(collection).commitAfterAdd(commitAfterAdd).build();
  }

  private void createStore(IndexType indexType) {
    createStore(indexType, true);
  }

  private static float[] vec(float a, float b, float c, float d) {
    return new float[] {a, b, c, d};
  }

  @Nested
  @Tag("unit")
  class AddSingleEmbedding {

    @Test
    void addEmbeddingReturnsId() {
      createStore(IndexType.FLAT);
      Embedding emb = Embedding.from(vec(1, 0, 0, 0));
      String id = store.add(emb);

      assertThat(id).isNotNull().isNotEmpty();
    }

    @Test
    void addWithExplicitId() {
      createStore(IndexType.FLAT);
      Embedding emb = Embedding.from(vec(1, 0, 0, 0));
      store.add("custom-id", emb);

      EmbeddingSearchResult<TextSegment> result =
          store.search(EmbeddingSearchRequest.builder().queryEmbedding(emb).maxResults(1).build());

      assertThat(result.matches()).hasSize(1);
      assertThat(result.matches().getFirst().embeddingId()).isEqualTo("custom-id");
    }

    @Test
    void addWithTextSegment() {
      createStore(IndexType.FLAT);
      Embedding emb = Embedding.from(vec(1, 0, 0, 0));
      TextSegment segment = TextSegment.from("Hello world", Metadata.from(Map.of("key", "val")));
      String id = store.add(emb, segment);

      assertThat(id).isNotNull();

      EmbeddingSearchResult<TextSegment> result =
          store.search(EmbeddingSearchRequest.builder().queryEmbedding(emb).maxResults(1).build());

      assertThat(result.matches()).hasSize(1);
      EmbeddingMatch<TextSegment> match = result.matches().getFirst();
      assertThat(match.embedded()).isNotNull();
      assertThat(match.embedded().text()).isEqualTo("Hello world");
      assertThat(match.embedded().metadata().getString("key")).isEqualTo("val");
    }

    @Test
    void addEmbeddingOnlyReturnsNullSegment() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());

      assertThat(result.matches()).hasSize(1);
      // No TextSegment was provided, so embedded() should be null
      assertThat(result.matches().getFirst().embedded()).isNull();
    }
  }

  @Nested
  @Tag("unit")
  class AddBatch {

    @Test
    void addAllEmbeddings() {
      createStore(IndexType.FLAT);
      List<Embedding> embeddings =
          List.of(
              Embedding.from(vec(1, 0, 0, 0)),
              Embedding.from(vec(0, 1, 0, 0)),
              Embedding.from(vec(0, 0, 1, 0)));
      List<String> ids = store.addAll(embeddings);

      assertThat(ids).hasSize(3);
      assertThat(collection.size()).isEqualTo(3);
    }

    @Test
    void addAllEmbeddingsWithSegments() {
      createStore(IndexType.FLAT);
      List<Embedding> embeddings =
          List.of(Embedding.from(vec(1, 0, 0, 0)), Embedding.from(vec(0, 1, 0, 0)));
      List<TextSegment> segments = List.of(TextSegment.from("alpha"), TextSegment.from("beta"));
      List<String> ids = store.addAll(embeddings, segments);

      assertThat(ids).hasSize(2);

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());

      assertThat(result.matches()).hasSize(2);
    }

    @Test
    void addAllWithExplicitIds() {
      createStore(IndexType.FLAT);
      List<String> ids = List.of("id-a", "id-b");
      List<Embedding> embeddings =
          List.of(Embedding.from(vec(1, 0, 0, 0)), Embedding.from(vec(0, 1, 0, 0)));
      List<TextSegment> segments = List.of(TextSegment.from("alpha"), TextSegment.from("beta"));
      store.addAll(ids, embeddings, segments);

      assertThat(collection.size()).isEqualTo(2);
    }
  }

  @Nested
  @Tag("unit")
  class Search {

    @Test
    void searchReturnsMatchesWithScores() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("near"));
      store.add(Embedding.from(vec(0, 0, 0, 1)), TextSegment.from("far"));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());

      assertThat(result.matches()).hasSize(2);
      assertThat(result.matches().getFirst().score()).isGreaterThan(0.0);
    }

    @Test
    void resultsReturnedInDescendingScoreOrder() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("exact"));
      store.add(Embedding.from(vec(0.7f, 0.7f, 0, 0)), TextSegment.from("close"));
      store.add(Embedding.from(vec(0, 0, 0, 1)), TextSegment.from("far"));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(3)
                  .build());

      assertThat(result.matches()).hasSize(3);
      List<EmbeddingMatch<TextSegment>> matches = result.matches();
      for (int i = 0; i < matches.size() - 1; i++) {
        assertThat(matches.get(i).score())
            .as("match[%d].score >= match[%d].score", i, i + 1)
            .isGreaterThanOrEqualTo(matches.get(i + 1).score());
      }
    }

    @Test
    void maxResultsLimitsOutput() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)));
      store.add(Embedding.from(vec(0, 1, 0, 0)));
      store.add(Embedding.from(vec(0, 0, 1, 0)));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());

      assertThat(result.matches()).hasSize(2);
    }

    @Test
    void minScoreFilters() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("exact"));
      store.add(Embedding.from(vec(0, 0, 0, 1)), TextSegment.from("orthogonal"));

      EmbeddingSearchResult<TextSegment> highThreshold =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(10)
                  .minScore(0.99)
                  .build());

      EmbeddingSearchResult<TextSegment> lowThreshold =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(10)
                  .minScore(0.0)
                  .build());

      assertThat(highThreshold.matches().size()).isLessThanOrEqualTo(lowThreshold.matches().size());
    }

    @Test
    void searchReturnsEmbeddingVectors() {
      createStore(IndexType.FLAT);
      float[] vec = vec(1, 0, 0, 0);
      store.add(Embedding.from(vec));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec))
                  .maxResults(1)
                  .build());

      assertThat(result.matches()).hasSize(1);
      assertThat(result.matches().getFirst().embedding()).isNotNull();
    }
  }

  @Nested
  @Tag("unit")
  class MetadataPreservation {

    @Test
    void metadataRoundTrip() {
      createStore(IndexType.FLAT);
      Metadata md = new Metadata();
      md.put("str_key", "str_val");
      md.put("num_key", 42L);
      md.put("dbl_key", 3.14);

      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("metadata test", md));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());

      assertThat(result.matches()).hasSize(1);
      Metadata resultMd = result.matches().getFirst().embedded().metadata();
      assertThat(resultMd.getString("str_key")).isEqualTo("str_val");
      assertThat(resultMd.getLong("num_key")).isEqualTo(42L);
      assertThat(resultMd.getDouble("dbl_key")).isEqualTo(3.14);
    }
  }

  @Nested
  @Tag("unit")
  class RemoveOperations {

    @Test
    void remove_deletesDocument_notFoundInSearch() {
      createStore(IndexType.FLAT);
      Embedding emb = Embedding.from(vec(1, 0, 0, 0));
      store.add("to-delete", emb);

      store.remove("to-delete");

      EmbeddingSearchResult<TextSegment> result =
          store.search(EmbeddingSearchRequest.builder().queryEmbedding(emb).maxResults(10).build());
      assertThat(result.matches()).isEmpty();
      assertThat(collection.size()).isEqualTo(0);
    }

    @Test
    void remove_unknownId_noException() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)));

      // Should not throw for unknown id
      store.remove("nonexistent-id");

      assertThat(collection.size()).isEqualTo(1);
    }

    @Test
    void removeAll_byIds_deletesMultiple() {
      createStore(IndexType.FLAT);
      store.add("id-a", Embedding.from(vec(1, 0, 0, 0)));
      store.add("id-b", Embedding.from(vec(0, 1, 0, 0)));
      store.add("id-c", Embedding.from(vec(0, 0, 1, 0)));

      store.removeAll(List.of("id-a", "id-c"));

      assertThat(collection.size()).isEqualTo(1);
      assertThat(collection.contains("id-b")).isTrue();
      assertThat(collection.contains("id-a")).isFalse();
      assertThat(collection.contains("id-c")).isFalse();
    }

    @Test
    void removeAll_byFilter_deletesMatching() {
      createStore(IndexType.FLAT);
      store.add(
          Embedding.from(vec(1, 0, 0, 0)),
          TextSegment.from("keep", Metadata.from(Map.of("status", "active"))));
      store.add(
          Embedding.from(vec(0, 1, 0, 0)),
          TextSegment.from("remove1", Metadata.from(Map.of("status", "archived"))));
      store.add(
          Embedding.from(vec(0, 0, 1, 0)),
          TextSegment.from("remove2", Metadata.from(Map.of("status", "archived"))));

      store.removeAll(new IsEqualTo("status", "archived"));

      assertThat(collection.size()).isEqualTo(1);

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(10)
                  .build());
      assertThat(result.matches()).hasSize(1);
      assertThat(result.matches().getFirst().embedded().text()).isEqualTo("keep");
    }

    @Test
    void removeAll_clearsEverything() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)));
      store.add(Embedding.from(vec(0, 1, 0, 0)));
      store.add(Embedding.from(vec(0, 0, 1, 0)));

      store.removeAll();

      assertThat(collection.size()).isEqualTo(0);
    }

    @Test
    void remove_withCommitAfterAddFalse_requiresExplicitCommit() {
      createStore(IndexType.FLAT, false);
      store.add("doc-1", Embedding.from(vec(1, 0, 0, 0)));
      store.commit();

      store.remove("doc-1");
      // Without commitAfterAdd, deletion is staged but not yet committed
      assertThat(collection.size()).isEqualTo(1);

      store.commit();
      assertThat(collection.size()).isEqualTo(0);
    }
  }

  @Nested
  @Tag("unit")
  class CommitStrategy {

    @Test
    void commitAfterAddTrueGivesImmediateSearchability() {
      createStore(IndexType.FLAT, true);

      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("auto commit"));

      // Should be immediately searchable without explicit commit
      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());
      assertThat(result.matches()).isNotEmpty();
    }

    @Test
    void commitAfterAddFalseRequiresExplicitCommit() {
      createStore(IndexType.FLAT, false);

      store.add(Embedding.from(vec(1, 0, 0, 0)));

      // Before commit, search should return empty
      EmbeddingSearchResult<TextSegment> beforeCommit =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());
      assertThat(beforeCommit.matches()).isEmpty();

      store.commit();

      EmbeddingSearchResult<TextSegment> afterCommit =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());
      assertThat(afterCommit.matches()).isNotEmpty();
    }

    @Test
    void addAllWithCommitAfterAddTrue() {
      createStore(IndexType.FLAT, true);

      store.addAll(
          List.of(Embedding.from(vec(1, 0, 0, 0)), Embedding.from(vec(0, 1, 0, 0))),
          List.of(TextSegment.from("a"), TextSegment.from("b")));

      // Should be immediately searchable
      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());
      assertThat(result.matches()).hasSize(2);
    }
  }

  @Nested
  @Tag("unit")
  class IndexTypes {

    @Test
    void flatIndex() {
      createStore(IndexType.FLAT);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("flat test"));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(1)
                  .build());
      assertThat(result.matches()).hasSize(1);
    }

    @Test
    void hnswIndex() {
      createStore(IndexType.HNSW);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("hnsw alpha"));
      store.add(Embedding.from(vec(0, 1, 0, 0)), TextSegment.from("hnsw beta"));
      store.add(Embedding.from(vec(0, 0, 1, 0)), TextSegment.from("hnsw gamma"));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());
      assertThat(result.matches()).isNotEmpty();
      assertThat(result.matches().size()).isLessThanOrEqualTo(2);
    }

    @Test
    void vamanaIndex() {
      createStore(IndexType.VAMANA);
      store.add(Embedding.from(vec(1, 0, 0, 0)), TextSegment.from("vamana alpha"));
      store.add(Embedding.from(vec(0, 1, 0, 0)), TextSegment.from("vamana beta"));
      store.add(Embedding.from(vec(0, 0, 1, 0)), TextSegment.from("vamana gamma"));

      EmbeddingSearchResult<TextSegment> result =
          store.search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1, 0, 0, 0)))
                  .maxResults(2)
                  .build());
      assertThat(result.matches()).isNotEmpty();
      assertThat(result.matches().size()).isLessThanOrEqualTo(2);
    }
  }

  @Nested
  @Tag("unit")
  class Lifecycle {

    @Test
    void closeIsIdempotent() {
      createStore(IndexType.FLAT);
      store.close();
      store.close(); // should not throw
      store = null; // prevent double close in tearDown
    }
  }

  @Nested
  @Tag("unit")
  class MmrDiversity {
    // "a","b" near-duplicates + highly relevant; "c" diverse + moderately relevant. Query
    // ~[1,0,..].
    private void createMmr(Float lambda) {
      collection =
          VectorCollection.builder()
              .dimension(DIMENSION)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .autoCommitThreshold(Integer.MAX_VALUE)
              .build();
      var b = JavaVectorsEmbeddingStore.builder(collection).commitAfterAdd(true);
      if (lambda != null) {
        b.mmr(lambda, 4);
      }
      store = b.build();
      store.addAll(
          List.of("a", "b", "c"),
          List.of(
              Embedding.from(vec(1f, 0.05f, 0f, 0f)),
              Embedding.from(vec(1f, 0.06f, 0f, 0f)),
              Embedding.from(vec(0.6f, 0.8f, 0f, 0f))),
          List.of(TextSegment.from("a"), TextSegment.from("b"), TextSegment.from("c")));
    }

    private List<String> topIds(int maxResults) {
      return store
          .search(
              EmbeddingSearchRequest.builder()
                  .queryEmbedding(Embedding.from(vec(1f, 0f, 0f, 0f)))
                  .maxResults(maxResults)
                  .build())
          .matches()
          .stream()
          .map(EmbeddingMatch::embeddingId)
          .toList();
    }

    @Test
    void withoutMmr_top2AreTheRedundantPair() {
      createMmr(null);
      assertThat(topIds(2)).containsExactly("a", "b"); // relevance order; diverse c excluded
    }

    @Test
    void withMmr_diverseSegmentReplacesRedundantNeighbor() {
      createMmr(0.3f);
      assertThat(topIds(2)).containsExactly("a", "c"); // a, then diverse c; redundant b dropped
    }
  }
}
