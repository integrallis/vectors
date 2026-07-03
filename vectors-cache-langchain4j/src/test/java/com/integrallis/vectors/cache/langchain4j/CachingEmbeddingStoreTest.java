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
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
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

  @Test
  void allMutationOverloadsDelegateAndReturnDelegateResults() {
    Embedding embedding = Embedding.from(new float[] {0.5f, 0.5f, 0f});
    TextSegment payload = TextSegment.from("mixed");

    assertThat(store.delegate()).isSameAs(fake);
    assertThat(store.add(embedding)).isNotBlank();
    store.add("explicit", embedding);
    assertThat(store.add(embedding, payload)).isNotBlank();
    assertThat(store.addAll(List.of(embedding))).hasSize(1);
    assertThat(store.addAll(List.of(embedding), List.of(payload))).hasSize(1);
    store.addAll(List.of("bulk"), List.of(embedding), List.of(payload));

    store.remove("explicit");
    store.removeAll(List.of("bulk"));
    Filter allPayloads = ignored -> true;
    store.removeAll(allPayloads);

    EmbeddingSearchResult<TextSegment> result =
        store.search(
            EmbeddingSearchRequest.builder().queryEmbedding(embedding).maxResults(10).build());
    assertThat(result.matches()).isEmpty();
  }

  @Test
  void mutationCannotInstallStaleResultAfterInvalidation() throws Exception {
    BlockingSearchStore<TextSegment> blocking = new BlockingSearchStore<>();
    VectorCache<EmbeddingSearchRequest, EmbeddingSearchResult<TextSegment>> localCache =
        CaffeineVectorCache.<EmbeddingSearchRequest, EmbeddingSearchResult<TextSegment>>builder()
            .maximumSize(64)
            .build();
    CachingEmbeddingStore<TextSegment> localStore =
        new CachingEmbeddingStore<>(blocking, localCache);
    localStore.add(Embedding.from(new float[] {1f, 0f, 0f}), TextSegment.from("apple"));
    blocking.delegate.searchCalls.set(0);

    EmbeddingSearchRequest req =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(Embedding.from(new float[] {1f, 0f, 0f}))
            .maxResults(1)
            .build();
    blocking.blockSearch.set(true);
    ExecutorService executor = Executors.newFixedThreadPool(2);
    try {
      Future<EmbeddingSearchResult<TextSegment>> inFlightSearch =
          executor.submit(() -> localStore.search(req));
      assertThat(blocking.searchEntered.await(5, TimeUnit.SECONDS)).isTrue();

      Future<?> mutation = executor.submit(() -> localStore.removeAll());
      Thread.sleep(100);
      assertThat(mutation.isDone()).isFalse();

      blocking.releaseSearch.countDown();
      assertThat(inFlightSearch.get(5, TimeUnit.SECONDS).matches()).hasSize(1);
      mutation.get(5, TimeUnit.SECONDS);

      EmbeddingSearchResult<TextSegment> afterMutation = localStore.search(req);
      assertThat(afterMutation.matches()).isEmpty();
      assertThat(blocking.delegate.searchCalls.get()).isEqualTo(2);
    } finally {
      executor.shutdownNow();
    }
  }

  private static final class BlockingSearchStore<T> implements EmbeddingStore<T> {
    private final FakeEmbeddingStore<T> delegate = new FakeEmbeddingStore<>();
    private final AtomicBoolean blockSearch = new AtomicBoolean();
    private final CountDownLatch searchEntered = new CountDownLatch(1);
    private final CountDownLatch releaseSearch = new CountDownLatch(1);

    @Override
    public String add(Embedding embedding) {
      return delegate.add(embedding);
    }

    @Override
    public void add(String id, Embedding embedding) {
      delegate.add(id, embedding);
    }

    @Override
    public String add(Embedding embedding, T embedded) {
      return delegate.add(embedding, embedded);
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {
      return delegate.addAll(embeddings);
    }

    @Override
    public void removeAll() {
      delegate.removeAll();
    }

    @Override
    public EmbeddingSearchResult<T> search(EmbeddingSearchRequest request) {
      if (blockSearch.get()) {
        searchEntered.countDown();
        try {
          assertThat(releaseSearch.await(5, TimeUnit.SECONDS)).isTrue();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("interrupted while waiting to search", e);
        } finally {
          blockSearch.set(false);
        }
      }
      return delegate.search(request);
    }
  }
}
