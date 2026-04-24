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

  @Test
  void findRelevantUsesSearchCache() {
    Embedding q = Embedding.from(new float[] {1f, 0f, 0f});
    store.findRelevant(q, 1);
    store.findRelevant(q, 1);

    assertThat(fake.searchCalls.get()).isEqualTo(1);
  }
}
