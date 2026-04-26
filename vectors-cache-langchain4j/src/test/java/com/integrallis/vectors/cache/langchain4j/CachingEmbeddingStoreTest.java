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
package com.integrallis.vectors.cache.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CachingEmbeddingStoreTest {

  private FakeEmbeddingStore<TextSegment> fake;
  private CachingEmbeddingStore<TextSegment> store;
  private VectorCache<EmbeddingSearchRequest, EmbeddingSearchResult<TextSegment>> cache;

  @BeforeEach
  void setUp() {
    fake = new FakeEmbeddingStore<>();
    cache =
        CaffeineVectorCache.<EmbeddingSearchRequest, EmbeddingSearchResult<TextSegment>>builder()
            .maximumSize(64)
            .build();
    store = new CachingEmbeddingStore<>(fake, cache);

    store.add(Embedding.from(new float[] {1f, 0f, 0f}), TextSegment.from("apple"));
    store.add(Embedding.from(new float[] {0f, 1f, 0f}), TextSegment.from("banana"));
    store.add(Embedding.from(new float[] {0f, 0f, 1f}), TextSegment.from("cherry"));
    fake.searchCalls.set(0);
  }

  @Test
  void repeatedSearchWithSameRequestIsCached() {
    EmbeddingSearchRequest req =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {1f, 0f, 0f}))
            .maxResults(1)
            .build();

    EmbeddingSearchResult<TextSegment> first = store.search(req);
    EmbeddingSearchResult<TextSegment> second = store.search(req);

    assertThat(first.matches()).hasSize(1);
    assertThat(first.matches().get(0).embedded().text()).isEqualTo("apple");
    assertThat(second).isSameAs(first);
    assertThat(fake.searchCalls.get()).isEqualTo(1);
  }

  @Test
  void differentRequestsAreCachedIndependently() {
    EmbeddingSearchRequest reqA =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {1f, 0f, 0f}))
            .maxResults(1)
            .build();
    EmbeddingSearchRequest reqB =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {0f, 1f, 0f}))
            .maxResults(1)
            .build();

    store.search(reqA);
    store.search(reqB);
    store.search(reqA);
    store.search(reqB);

    assertThat(fake.searchCalls.get()).isEqualTo(2);
  }

  @Test
  void addInvalidatesSearchCache() {
    EmbeddingSearchRequest req =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {1f, 0f, 0f}))
            .maxResults(3)
            .build();

    store.search(req);
    store.add(Embedding.from(new float[] {0.9f, 0.1f, 0f}), TextSegment.from("apricot"));
    store.search(req);

    assertThat(fake.searchCalls.get()).isEqualTo(2);
  }

  @Test
  void removeInvalidatesSearchCache() {
    EmbeddingSearchRequest req =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {1f, 0f, 0f}))
            .maxResults(3)
            .build();

    store.search(req);
    store.removeAll();
    store.search(req);

    assertThat(fake.searchCalls.get()).isEqualTo(2);
  }
}
