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
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.vectorstore.EmbeddedDocument;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.util.MimeTypeUtils;

/**
 * The Spring AI 2.1 additions, exercised the way an application reaches them: through the {@link
 * VectorStore} interface, with Spring AI's own {@link EmbeddedDocument}, and with an observation
 * registry attached. A test that called the concrete class with a stand-in record passed while the
 * store bypassed the framework's observed template methods (0.1.44); these cannot.
 */
@Tag("unit")
class SpringAi21ContractTest {

  private EmbeddingModel embeddingModel;
  private VectorCollection collection;
  private VectorStore store;
  private final List<String> observed = new ArrayList<>();

  @BeforeEach
  void setUp() {
    embeddingModel = mock(EmbeddingModel.class);
    when(embeddingModel.dimensions()).thenReturn(4);
    ObservationRegistry registry = ObservationRegistry.create();
    registry
        .observationConfig()
        .observationHandler(
            new ObservationHandler<Observation.Context>() {
              @Override
              public boolean supportsContext(Observation.Context context) {
                return true;
              }

              @Override
              public void onStart(Observation.Context context) {
                observed.add(context.getName());
              }
            });
    collection =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .autoCommitThreshold(Integer.MAX_VALUE)
            .build();
    store =
        JavaVectorsVectorStore.builder(embeddingModel, collection)
            .observationRegistry(registry)
            .batchingStrategy(docs -> List.of(docs))
            .build();
  }

  @AfterEach
  void tearDown() {
    collection.close();
  }

  private static EmbeddedDocument embedded(String id, String text, float... vector) {
    return new EmbeddedDocument(new Document(id, text, Map.of("k", id)), vector);
  }

  private static SearchRequest top(int k) {
    return SearchRequest.builder().query("unused").topK(k).build();
  }

  private void verifyEmbedderNeverAskedToEmbed() {
    verify(embeddingModel, never()).embed(any(String.class));
    verify(embeddingModel, never()).embed(any(Document.class));
    verify(embeddingModel, never())
        .embed(anyList(), any(EmbeddingOptions.class), any(BatchingStrategy.class));
  }

  @Test
  void upsertStoresTheSuppliedVectorWithoutCallingTheEmbedder() {
    store.upsert(List.of(embedded("a", "alpha", 1, 0, 0, 0), embedded("b", "beta", 0, 1, 0, 0)));

    List<Document> hits = store.similaritySearch(new float[] {0, 1, 0, 0}, top(1));

    assertThat(hits).extracting(Document::getId).containsExactly("b");
    assertThat(hits.get(0).getText()).isEqualTo("beta");
    assertThat(hits.get(0).getMetadata()).containsEntry("k", "b");
    verifyEmbedderNeverAskedToEmbed();
  }

  @Test
  void upsertReplacesAnExistingIdRatherThanAddingASecondCopy() {
    store.upsert(List.of(embedded("a", "first", 1, 0, 0, 0)));
    store.upsert(List.of(embedded("a", "second", 0, 1, 0, 0)));

    List<Document> hits = store.similaritySearch(new float[] {0, 1, 0, 0}, top(10));

    assertThat(hits).hasSize(1);
    assertThat(hits.get(0).getText()).isEqualTo("second");
  }

  @Test
  void upsertIsObserved() {
    store.upsert(List.of(embedded("a", "alpha", 1, 0, 0, 0)));

    assertThat(observed).hasSize(1);
  }

  @Test
  void queryVectorSearchIsObserved() {
    store.upsert(List.of(embedded("a", "alpha", 1, 0, 0, 0)));
    observed.clear();

    store.similaritySearch(new float[] {1, 0, 0, 0}, top(1));

    assertThat(observed).hasSize(1);
  }

  @Test
  void queryVectorSearchHonoursTopKAndFilter() {
    store.upsert(
        List.of(
            embedded("a", "alpha", 1, 0, 0, 0),
            embedded("b", "beta", 0.9f, 0.1f, 0, 0),
            embedded("c", "gamma", 0, 1, 0, 0)));

    assertThat(store.similaritySearch(new float[] {1, 0, 0, 0}, top(1)))
        .extracting(Document::getId)
        .containsExactly("a");
    assertThat(
            store.similaritySearch(
                new float[] {1, 0, 0, 0},
                SearchRequest.builder().query("unused").topK(10).filterExpression("k == 'c'").build()))
        .extracting(Document::getId)
        .containsExactly("c");
    verifyEmbedderNeverAskedToEmbed();
  }

  @Test
  void upsertRejectsANonTextDocumentAsTheFrameworkRequires() {
    Document image =
        Document.builder()
            .id("img")
            .media(Media.builder().mimeType(MimeTypeUtils.IMAGE_PNG).data(new byte[] {1}).build())
            .build();

    assertThatThrownBy(
            () -> store.upsert(List.of(new EmbeddedDocument(image, new float[] {1, 0, 0, 0}))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Only text documents");
  }

  @Test
  void aVectorOfTheWrongDimensionIsRejectedOnWriteAndOnQuery() {
    assertThatThrownBy(() -> store.upsert(List.of(embedded("w", "wide", new float[8]))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimension");
    assertThatThrownBy(() -> store.similaritySearch(new float[] {1, 0, 0, 0, 0, 0, 0, 0}, top(1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimension");
  }

  @Test
  void aBatchWithOneBadVectorWritesNothing() {
    assertThatThrownBy(
            () ->
                store.upsert(
                    List.of(embedded("good", "fine", 1, 0, 0, 0), embedded("bad", "wide", new float[8]))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("'bad'");
    collection.commit();

    assertThat(store.similaritySearch(new float[] {1, 0, 0, 0}, top(10))).isEmpty();
  }

  @Test
  void theHookRefusesAnythingThatIsNotAnEmbeddedDocument() {
    JavaVectorsVectorStore concrete = (JavaVectorsVectorStore) store;

    assertThatThrownBy(() -> concrete.doUpsert(List.of("not an embedded document")))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("EmbeddedDocument")
        .hasMessageContaining("java.lang.String");
    List<Object> withNull = new ArrayList<>();
    withNull.add(null);
    assertThatThrownBy(() -> concrete.doUpsert(withNull))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("got null");
    concrete.doUpsert(List.of());
    assertThat(store.similaritySearch(new float[] {1, 0, 0, 0}, top(10))).isEmpty();
  }
}
