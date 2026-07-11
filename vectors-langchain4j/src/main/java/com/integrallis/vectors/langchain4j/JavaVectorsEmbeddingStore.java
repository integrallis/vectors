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

import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.hybrid.MaximalMarginalRelevance;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * LangChain4j {@link EmbeddingStore} backed by java-vectors.
 *
 * <p>Drop-in replacement for LangChain4j's {@code InMemoryEmbeddingStore} with HNSW/Vamana graph
 * indexing, SIMD-accelerated distance kernels, optional quantization (SQ, PQ, BQ, RaBitQ, NVQ), and
 * mmap persistence.
 *
 * <p>Unlike the Spring AI adapter, LangChain4j passes pre-computed embeddings, so no embedding
 * model is needed here.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorCollection collection = VectorCollection.builder()
 *     .dimension(384)
 *     .metric(SimilarityFunction.COSINE)
 *     .indexType(IndexType.HNSW)
 *     .quantizer(QuantizerKind.SQ8)
 *     .autoCommitThreshold(100)
 *     .build();
 *
 * EmbeddingStore<TextSegment> store = JavaVectorsEmbeddingStore.builder(collection)
 *     .commitAfterAdd(true)
 *     .build();
 * }</pre>
 */
public class JavaVectorsEmbeddingStore implements EmbeddingStore<TextSegment>, AutoCloseable {

  private final VectorCollection collection;
  private final boolean commitAfterAdd;
  private final Float mmrLambda; // null = MMR disabled
  private final int mmrFetchMultiplier;

  private JavaVectorsEmbeddingStore(Builder builder) {
    this.collection = builder.collection;
    this.commitAfterAdd = builder.commitAfterAdd;
    this.mmrLambda = builder.mmrLambda;
    this.mmrFetchMultiplier = builder.mmrFetchMultiplier;
  }

  /**
   * Creates a builder for {@link JavaVectorsEmbeddingStore}.
   *
   * @param collection the pre-configured vector collection
   * @return a new builder
   */
  public static Builder builder(VectorCollection collection) {
    return new Builder(collection);
  }

  @Override
  public String add(Embedding embedding) {
    String id = generateId();
    add(id, embedding);
    return id;
  }

  @Override
  public void add(String id, Embedding embedding) {
    collection.add(
        new com.integrallis.vectors.core.Document(id, embedding.vector(), null, Map.of()));
    commitIfNeeded();
  }

  @Override
  public String add(Embedding embedding, TextSegment textSegment) {
    String id = generateId();
    Map<String, MetadataValue> metadata =
        textSegment != null ? MetadataConverter.toJavaVectors(textSegment.metadata()) : Map.of();
    String text = textSegment != null ? textSegment.text() : null;
    collection.add(
        new com.integrallis.vectors.core.Document(id, embedding.vector(), text, metadata));
    commitIfNeeded();
    return id;
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings) {
    List<String> ids = newIds(embeddings.size());
    List<com.integrallis.vectors.core.Document> docs = new ArrayList<>(embeddings.size());
    for (int i = 0; i < embeddings.size(); i++) {
      docs.add(
          new com.integrallis.vectors.core.Document(
              ids.get(i), embeddings.get(i).vector(), null, Map.of()));
    }
    collection.addAll(docs);
    commitIfNeeded();
    return ids;
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings, List<TextSegment> segments) {
    List<String> ids = newIds(embeddings.size());
    addAll(ids, embeddings, segments);
    return ids;
  }

  @Override
  public void addAll(List<String> ids, List<Embedding> embeddings, List<TextSegment> segments) {
    List<com.integrallis.vectors.core.Document> docs = new ArrayList<>(ids.size());
    for (int i = 0; i < ids.size(); i++) {
      TextSegment segment = segments != null && i < segments.size() ? segments.get(i) : null;
      String text = segment != null ? segment.text() : null;
      Map<String, MetadataValue> metadata =
          segment != null ? MetadataConverter.toJavaVectors(segment.metadata()) : Map.of();
      docs.add(
          new com.integrallis.vectors.core.Document(
              ids.get(i), embeddings.get(i).vector(), text, metadata));
    }
    collection.addAll(docs);
    commitIfNeeded();
  }

  @Override
  public void remove(String id) {
    collection.delete(id);
    commitIfNeeded();
  }

  @Override
  public void removeAll(Collection<String> ids) {
    for (String id : ids) {
      collection.delete(id);
    }
    commitIfNeeded();
  }

  @Override
  public void removeAll(dev.langchain4j.store.embedding.filter.Filter filter) {
    collection.deleteWhere(FilterConverter.convert(filter));
    commitIfNeeded();
  }

  @Override
  public void removeAll() {
    collection.deleteWhere(new com.integrallis.vectors.core.filter.Filter.All());
    commitIfNeeded();
  }

  @Override
  public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest request) {
    float[] query = request.queryEmbedding().vector();
    int maxResults = request.maxResults();
    boolean useMmr = mmrLambda != null;
    // With MMR on, over-fetch a larger candidate pool, then diversify down to maxResults.
    int fetch = useMmr ? Math.multiplyExact(maxResults, mmrFetchMultiplier) : maxResults;

    var jvRequestBuilder = com.integrallis.vectors.db.SearchRequest.builder(query, fetch);

    if (request.minScore() > 0.0) {
      jvRequestBuilder.minScore((float) request.minScore());
    }

    if (request.filter() != null) {
      jvRequestBuilder.filter(FilterConverter.convert(request.filter()));
    }

    com.integrallis.vectors.db.SearchResult jvResult = collection.search(jvRequestBuilder.build());
    List<com.integrallis.vectors.db.SearchResult.Hit> hits = jvResult.hits();

    if (useMmr && hits.size() > maxResults) {
      hits = applyMmr(query, hits, maxResults);
    }

    List<EmbeddingMatch<TextSegment>> matches = new ArrayList<>(hits.size());
    for (com.integrallis.vectors.db.SearchResult.Hit hit : hits) {
      com.integrallis.vectors.core.Document jvDoc = hit.document();

      TextSegment segment = null;
      if (jvDoc.text() != null) {
        Metadata lc4jMetadata = MetadataConverter.toLangChain4j(jvDoc.metadata());
        segment = TextSegment.from(jvDoc.text(), lc4jMetadata);
      }

      Embedding embedding = jvDoc.vector() != null ? Embedding.from(jvDoc.vector()) : null;

      matches.add(new EmbeddingMatch<>((double) hit.score(), hit.id(), embedding, segment));
    }
    return new EmbeddingSearchResult<>(matches);
  }

  private List<com.integrallis.vectors.db.SearchResult.Hit> applyMmr(
      float[] query, List<com.integrallis.vectors.db.SearchResult.Hit> hits, int maxResults) {
    float[][] candidateVectors = new float[hits.size()][];
    for (int i = 0; i < hits.size(); i++) {
      candidateVectors[i] = hits.get(i).document().vector();
      if (candidateVectors[i] == null) {
        // Vectors not returned (e.g. stripped by config) — fall back to relevance order.
        return hits.subList(0, maxResults);
      }
    }
    int[] selected =
        MaximalMarginalRelevance.select(
            query, candidateVectors, maxResults, mmrLambda, collection.config().metric());
    List<com.integrallis.vectors.db.SearchResult.Hit> reranked = new ArrayList<>(selected.length);
    for (int idx : selected) {
      reranked.add(hits.get(idx));
    }
    return reranked;
  }

  /**
   * Commits staged documents, making them searchable.
   *
   * <p>Call this after adding documents if {@code commitAfterAdd} is {@code false} and the
   * collection's auto-commit threshold has not been reached.
   */
  public void commit() {
    collection.commit();
  }

  @Override
  public void close() {
    collection.close();
  }

  private void commitIfNeeded() {
    if (commitAfterAdd) {
      collection.commit();
    }
  }

  private static String generateId() {
    return UUID.randomUUID().toString();
  }

  private static List<String> newIds(int count) {
    List<String> ids = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      ids.add(generateId());
    }
    return ids;
  }

  /** Builder for {@link JavaVectorsEmbeddingStore}. */
  public static final class Builder {

    private final VectorCollection collection;
    private boolean commitAfterAdd = true;
    private Float mmrLambda = null; // null = MMR disabled
    private int mmrFetchMultiplier = 4;

    private Builder(VectorCollection collection) {
      this.collection = Objects.requireNonNull(collection, "collection must not be null");
    }

    /**
     * Controls whether the adapter calls {@code collection.commit()} after each add/addAll
     * operation. Default: {@code true}.
     *
     * <p>When {@code true}, documents are immediately searchable after adding. When {@code false},
     * the caller must invoke {@code commit()} explicitly.
     */
    public Builder commitAfterAdd(boolean commitAfterAdd) {
      this.commitAfterAdd = commitAfterAdd;
      return this;
    }

    /**
     * Enables Maximal Marginal Relevance diversity re-ranking. {@code search()} over-fetches {@code
     * maxResults * fetchMultiplier} candidates and re-ranks them down to {@code maxResults}
     * diverse-but-relevant results using the collection's metric — so near-duplicate segments don't
     * crowd the LLM context window. Disabled by default.
     *
     * @param lambda relevance/diversity trade-off in {@code [0, 1]} (1 = pure relevance, 0 = max
     *     diversity; typical RAG values 0.5–0.7)
     */
    public Builder mmr(float lambda) {
      return mmr(lambda, this.mmrFetchMultiplier);
    }

    /**
     * Enables MMR with an explicit over-fetch multiplier (pool = {@code maxResults * multiplier}).
     */
    public Builder mmr(float lambda, int fetchMultiplier) {
      if (lambda < 0f || lambda > 1f) {
        throw new IllegalArgumentException("mmr lambda must be in [0, 1], got " + lambda);
      }
      if (fetchMultiplier < 1) {
        throw new IllegalArgumentException(
            "mmr fetchMultiplier must be >= 1, got " + fetchMultiplier);
      }
      this.mmrLambda = lambda;
      this.mmrFetchMultiplier = fetchMultiplier;
      return this;
    }

    /** Builds the embedding store. */
    public JavaVectorsEmbeddingStore build() {
      return new JavaVectorsEmbeddingStore(this);
    }
  }
}
