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
package com.integrallis.vectors.vcr.semantic;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * {@link CassetteStore} that supports both exact and semantic (similarity-based) lookup.
 *
 * <p>Writes fan out to two backing stores: an {@link ExactCassetteStore} for byte-identical
 * retrieval by key, and a {@link VectorCollection} that indexes the cassette's associated embedding
 * for nearest-neighbour search via {@link #retrieveSimilar(float[], float)}.
 *
 * <p>Intended for tests that want cache hits across prompts that are semantically equivalent but
 * not byte-identical (e.g. minor whitespace variations). Falls back to the exact store for keys
 * that have no associated embedding (e.g. chat cassettes).
 */
public final class SemanticCassetteStore implements CassetteStore {

  /** Default cosine-similarity threshold above which a semantic hit is accepted. */
  public static final float DEFAULT_SIMILARITY_THRESHOLD = 0.95f;

  private static final String METADATA_KEY = "cassette_key";

  private final ExactCassetteStore exactStore;
  private final VectorCollection index;
  private final float defaultThreshold;

  /**
   * Creates a store with the default similarity threshold ({@value #DEFAULT_SIMILARITY_THRESHOLD}).
   *
   * @param backend the backing storage for exact payloads
   * @param serializer the cassette serializer
   * @param dimension the embedding dimension used to size the vector index
   */
  public SemanticCassetteStore(
      StorageBackend backend, CassetteSerializer serializer, int dimension) {
    this(backend, serializer, dimension, DEFAULT_SIMILARITY_THRESHOLD);
  }

  /**
   * Creates a store with a custom threshold.
   *
   * @param backend the backing storage
   * @param serializer the cassette serializer
   * @param dimension the embedding dimension
   * @param defaultThreshold similarity threshold in {@code [0,1]} for accepting a semantic hit
   */
  public SemanticCassetteStore(
      StorageBackend backend,
      CassetteSerializer serializer,
      int dimension,
      float defaultThreshold) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    if (defaultThreshold < 0f || defaultThreshold > 1f) {
      throw new IllegalArgumentException("defaultThreshold must be in [0, 1]: " + defaultThreshold);
    }
    this.exactStore =
        new ExactCassetteStore(Objects.requireNonNull(backend, "backend"), serializer);
    this.defaultThreshold = defaultThreshold;
    this.index =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .autoCommitThreshold(1)
            .build();
  }

  @Override
  public void store(CassetteKey key, CassetteRecord record) {
    exactStore.store(key, record);
    float[] vector = primaryEmbedding(record);
    if (vector != null) {
      String docId = key.serializedKey();
      Map<String, MetadataValue> metadata = new HashMap<>();
      metadata.put(METADATA_KEY, MetadataValue.of(key.serializedKey()));
      Document doc = new Document(docId, vector, null, metadata);
      if (index.contains(docId)) {
        index.upsert(doc);
      } else {
        index.add(doc);
      }
      index.commit();
    }
  }

  @Override
  public Optional<CassetteRecord> retrieve(CassetteKey key) {
    return exactStore.retrieve(key);
  }

  @Override
  public boolean exists(CassetteKey key) {
    return exactStore.exists(key);
  }

  @Override
  public void delete(CassetteKey key) {
    String serialized = key.serializedKey();
    if (index.contains(serialized)) {
      index.delete(serialized);
      index.commit();
    }
    exactStore.delete(key);
  }

  @Override
  public List<CassetteKey> listByTestId(String testId) {
    return exactStore.listByTestId(testId);
  }

  /**
   * Finds the most similar cassette whose associated embedding exceeds the default threshold.
   *
   * @param queryEmbedding the query vector
   * @return the matching record, or empty if no hit meets the threshold
   */
  public Optional<CassetteRecord> retrieveSimilar(float[] queryEmbedding) {
    return retrieveSimilar(queryEmbedding, defaultThreshold);
  }

  /**
   * Finds the most similar cassette whose cosine similarity to {@code queryEmbedding} is at least
   * {@code threshold}.
   *
   * @param queryEmbedding the query vector
   * @param threshold cosine-similarity threshold in {@code [0, 1]}
   * @return the matching record, or empty if no hit meets the threshold
   */
  public Optional<CassetteRecord> retrieveSimilar(float[] queryEmbedding, float threshold) {
    Objects.requireNonNull(queryEmbedding, "queryEmbedding");
    if (index.size() == 0) {
      return Optional.empty();
    }
    SearchResult result = index.search(SearchRequest.builder(queryEmbedding, 1).build());
    if (result.hits().isEmpty()) {
      return Optional.empty();
    }
    SearchResult.Hit top = result.hits().get(0);
    if (top.score() < threshold) {
      return Optional.empty();
    }
    CassetteKey parsed = CassetteKey.parse(top.id());
    if (parsed == null) {
      return Optional.empty();
    }
    return exactStore.retrieve(parsed);
  }

  @Override
  public void close() throws IOException {
    try {
      index.close();
    } finally {
      exactStore.close();
    }
  }

  private static float[] primaryEmbedding(CassetteRecord record) {
    if (record instanceof CassetteRecord.Embedding e) {
      return e.embedding();
    }
    if (record instanceof CassetteRecord.BatchEmbedding b && b.embeddings().length > 0) {
      return b.embeddings()[0];
    }
    return null;
  }

  /**
   * @return the default similarity threshold
   */
  public float defaultThreshold() {
    return defaultThreshold;
  }
}
