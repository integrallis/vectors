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
package com.integrallis.vectors.cache;

import java.util.Optional;

/**
 * Similarity-search cache: a cache that returns a stored value whose associated embedding is closer
 * than a threshold to the lookup embedding.
 *
 * <p>Use-case: caching LLM prompt/response pairs where near-duplicate prompts should reuse the
 * previous response. The 0.92 default cosine threshold follows common practice in chat-assistant
 * caching and is a reasonable starting point for MiniLM-class embedding models.
 *
 * <p>Implementations typically delegate to a {@link com.integrallis.vectors.db.VectorCollection
 * VectorCollection} for the nearest-neighbour search and an {@link VectorCache} for the payload
 * store.
 *
 * @param <V> payload type (e.g. LLM response text, structured assistant output)
 */
public interface SemanticCache<V> extends AutoCloseable {

  /**
   * Exact lookup by {@code key}. Returns empty if the key is not cached (equivalent to {@link
   * VectorCache#get}).
   */
  Optional<V> get(String key);

  /** Exact insert or replace. */
  void put(String key, float[] embedding, V value);

  /**
   * Nearest-neighbour lookup. Returns the value whose embedding scores above {@link #threshold()}
   * against {@code queryEmbedding} under the configured similarity, or empty otherwise.
   *
   * @param queryEmbedding the lookup embedding
   * @return a cache hit, if any
   */
  Optional<Hit<V>> lookup(float[] queryEmbedding);

  /** Removes the entry by exact key. */
  void invalidate(String key);

  /** Removes every entry. */
  void invalidateAll();

  /** Snapshot of the lifetime counters. */
  CacheStats stats();

  /** The cosine / dot / Euclidean similarity threshold above which a lookup is considered a hit. */
  double threshold();

  /** The admission policy used to gate {@link #put} calls. */
  CacheAdmissionPolicy<V> admissionPolicy();

  /** Closes any underlying resources; default is no-op. */
  @Override
  default void close() {
    // no-op by default; implementations that own a VectorCollection should close it here.
  }

  /**
   * A semantic-cache hit.
   *
   * @param value the stored payload
   * @param score similarity score between the lookup embedding and the matched entry
   * @param <V> payload type
   */
  record Hit<V>(V value, double score) {}
}
