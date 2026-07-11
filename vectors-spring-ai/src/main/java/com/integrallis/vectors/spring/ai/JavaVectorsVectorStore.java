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

import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.hybrid.MaximalMarginalRelevance;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;

/**
 * Spring AI {@link org.springframework.ai.vectorstore.VectorStore} backed by java-vectors.
 *
 * <p>Drop-in replacement for Spring AI's {@code SimpleVectorStore} with HNSW/Vamana graph indexing,
 * SIMD-accelerated distance kernels, optional quantization (SQ, PQ, BQ, RaBitQ, NVQ), and mmap
 * persistence.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorCollection collection = VectorCollection.builder()
 *     .dimension(embeddingModel.dimensions())
 *     .metric(SimilarityFunction.COSINE)
 *     .indexType(IndexType.HNSW)
 *     .quantizer(QuantizerKind.SQ8)
 *     .autoCommitThreshold(1)
 *     .build();
 *
 * VectorStore store = JavaVectorsVectorStore.builder(embeddingModel, collection)
 *     .collectionName("my-docs")
 *     .build();
 * }</pre>
 */
public class JavaVectorsVectorStore extends AbstractObservationVectorStore
    implements AutoCloseable {

  static final String DATABASE_SYSTEM = "java-vectors";

  private final VectorCollection collection;
  private final String collectionName;
  private final boolean commitAfterAdd;
  private final Float mmrLambda; // null = MMR disabled
  private final int mmrFetchMultiplier;

  private JavaVectorsVectorStore(Builder builder) {
    super(builder);
    this.collection = builder.collection;
    this.collectionName = builder.collectionName;
    this.commitAfterAdd = builder.commitAfterAdd;
    this.mmrLambda = builder.mmrLambda;
    this.mmrFetchMultiplier = builder.mmrFetchMultiplier;
  }

  /**
   * Creates a builder for {@link JavaVectorsVectorStore}.
   *
   * @param embeddingModel the embedding model for vectorizing documents and queries
   * @param collection the java-vectors collection (pre-configured with index type, quantizer, etc.)
   * @return a new builder
   */
  public static Builder builder(EmbeddingModel embeddingModel, VectorCollection collection) {
    return new Builder(embeddingModel, collection);
  }

  @Override
  public void doAdd(List<Document> documents) {
    if (documents == null || documents.isEmpty()) {
      return;
    }

    List<com.integrallis.vectors.core.Document> jvDocs = new ArrayList<>(documents.size());
    for (Document doc : documents) {
      float[] embedding = this.embeddingModel.embed(doc);
      Map<String, MetadataValue> jvMetadata = MetadataConverter.toJavaVectors(doc.getMetadata());
      jvDocs.add(
          new com.integrallis.vectors.core.Document(
              doc.getId(), embedding, doc.getText(), jvMetadata));
    }
    collection.addAll(jvDocs);

    if (commitAfterAdd) {
      collection.commit();
    }
  }

  @Override
  public void doDelete(List<String> idList) {
    if (idList == null || idList.isEmpty()) {
      return;
    }
    for (String id : idList) {
      collection.delete(id);
    }
    if (commitAfterAdd) {
      collection.commit();
    }
  }

  @Override
  public List<Document> doSimilaritySearch(SearchRequest request) {
    float[] queryEmbedding = this.embeddingModel.embed(request.getQuery());
    int topK = request.getTopK();
    boolean useMmr = mmrLambda != null;
    // With MMR on, over-fetch a larger candidate pool, then diversify down to topK.
    int fetch = useMmr ? Math.multiplyExact(topK, mmrFetchMultiplier) : topK;

    var jvRequestBuilder = com.integrallis.vectors.db.SearchRequest.builder(queryEmbedding, fetch);

    if (request.getSimilarityThreshold() > 0.0) {
      jvRequestBuilder.minScore((float) request.getSimilarityThreshold());
    }

    if (request.hasFilterExpression()) {
      jvRequestBuilder.filter(FilterConverter.convert(request.getFilterExpression()));
    }

    com.integrallis.vectors.db.SearchResult jvResult = collection.search(jvRequestBuilder.build());
    List<com.integrallis.vectors.db.SearchResult.Hit> hits = jvResult.hits();

    if (useMmr && hits.size() > topK) {
      hits = applyMmr(queryEmbedding, hits, topK);
    }

    List<Document> results = new ArrayList<>(hits.size());
    for (com.integrallis.vectors.db.SearchResult.Hit hit : hits) {
      com.integrallis.vectors.core.Document jvDoc = hit.document();
      Map<String, Object> springMetadata = MetadataConverter.toSpringAi(jvDoc.metadata());

      Document springDoc =
          Document.builder()
              .id(hit.id())
              .text(jvDoc.text())
              .metadata(springMetadata)
              .score((double) hit.score())
              .build();
      results.add(springDoc);
    }
    return results;
  }

  private List<com.integrallis.vectors.db.SearchResult.Hit> applyMmr(
      float[] query, List<com.integrallis.vectors.db.SearchResult.Hit> hits, int topK) {
    float[][] candidateVectors = new float[hits.size()][];
    for (int i = 0; i < hits.size(); i++) {
      candidateVectors[i] = hits.get(i).document().vector();
      if (candidateVectors[i] == null) {
        // Vectors not returned (e.g. stripped by config) — fall back to relevance order.
        return hits.subList(0, topK);
      }
    }
    int[] selected =
        MaximalMarginalRelevance.select(
            query, candidateVectors, topK, mmrLambda, collection.config().metric());
    List<com.integrallis.vectors.db.SearchResult.Hit> reranked = new ArrayList<>(selected.length);
    for (int idx : selected) {
      reranked.add(hits.get(idx));
    }
    return reranked;
  }

  @Override
  public VectorStoreObservationContext.Builder createObservationContextBuilder(
      String operationName) {
    return VectorStoreObservationContext.builder(DATABASE_SYSTEM, operationName)
        .dimensions(collection.config().dimension())
        .collectionName(collectionName)
        .similarityMetric(collection.config().metric().name());
  }

  @Override
  public void close() {
    collection.close();
  }

  /** Builder for {@link JavaVectorsVectorStore}. */
  public static final class Builder extends AbstractVectorStoreBuilder<Builder> {

    private final VectorCollection collection;
    private String collectionName = "default";
    private boolean commitAfterAdd = true;
    private Float mmrLambda = null; // null = MMR disabled
    private int mmrFetchMultiplier = 4;

    private Builder(EmbeddingModel embeddingModel, VectorCollection collection) {
      super(embeddingModel);
      this.collection = Objects.requireNonNull(collection, "collection must not be null");
    }

    /** Sets the collection name used in observation context. Default: {@code "default"}. */
    public Builder collectionName(String collectionName) {
      this.collectionName =
          Objects.requireNonNull(collectionName, "collectionName must not be null");
      return this;
    }

    /**
     * Controls whether the adapter calls {@code collection.commit()} after each {@code doAdd()}.
     * Default: {@code true}.
     *
     * <p>When {@code true}, documents are immediately searchable after {@code add()}. When {@code
     * false}, the caller must invoke {@code collection.commit()} explicitly.
     */
    public Builder commitAfterAdd(boolean commitAfterAdd) {
      this.commitAfterAdd = commitAfterAdd;
      return this;
    }

    /**
     * Enables Maximal Marginal Relevance diversity re-ranking. The store over-fetches {@code topK *
     * fetchMultiplier} candidates and re-ranks them down to {@code topK} diverse-but-relevant
     * results using the collection's metric — so near-duplicate chunks don't crowd the LLM context
     * window. Disabled by default.
     *
     * @param lambda relevance/diversity trade-off in {@code [0, 1]} (1 = pure relevance, 0 = max
     *     diversity; typical RAG values 0.5–0.7)
     */
    public Builder mmr(float lambda) {
      return mmr(lambda, this.mmrFetchMultiplier);
    }

    /**
     * Enables MMR with an explicit over-fetch multiplier (candidate pool size = {@code topK *
     * fetchMultiplier}).
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

    @Override
    public JavaVectorsVectorStore build() {
      int modelDim = embeddingModel.dimensions();
      int collectionDim = collection.config().dimension();
      if (modelDim != collectionDim) {
        throw new IllegalArgumentException(
            "EmbeddingModel dimension ("
                + modelDim
                + ") does not match VectorCollection dimension ("
                + collectionDim
                + ")");
      }
      return new JavaVectorsVectorStore(this);
    }
  }
}
