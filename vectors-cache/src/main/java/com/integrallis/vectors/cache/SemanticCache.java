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

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Similarity-search cache: a cache that returns a stored value whose associated embedding is closer
 * than a threshold to the lookup embedding.
 *
 * <p>Use-case: caching LLM prompt/response pairs where near-duplicate prompts should reuse the
 * previous response. The 0.92 default cosine threshold follows common practice in chat-assistant
 * caching and is a reasonable starting point for MiniLM-class embedding models.
 *
 * <p>Implementations typically delegate to a vector collection for the nearest-neighbour search and
 * a {@link VectorCache} for the payload store.
 *
 * @param <V> payload type (e.g. LLM response text, structured assistant output)
 */
public interface SemanticCache<V> extends AutoCloseable {

  /** How many neighbours a default filtered lookup considers before giving up. */
  int FILTER_CANDIDATES = 16;

  /**
   * Exact lookup by {@code key}. Returns empty if the key is not cached (equivalent to {@link
   * VectorCache#get}).
   */
  Optional<V> get(String key);

  /** Exact insert or replace. */
  void put(String key, float[] embedding, V value);

  /**
   * Whether this cache stores and matches entry attributes.
   *
   * <p>Callers whose correctness depends on filtering — anything that must not serve an entry
   * produced under different conditions — should check this before accepting a cache, so the
   * mismatch surfaces at wiring time instead of on the first request.
   *
   * @return true if attributes are stored and matched
   */
  default boolean supportsAttributes() {
    return false;
  }

  /**
   * Stores a value along with the conditions it was produced under.
   *
   * <p>Attributes are what {@link CacheFilter} matches on: the model that generated the value, the
   * sampling options, the tenant. Implementations that cannot store them must reject a non-empty
   * map rather than drop it, because a filtered lookup over entries with missing attributes serves
   * exactly the wrong answers it was added to prevent.
   *
   * @param key exact cache key
   * @param embedding embedding associated with the value
   * @param value payload to cache
   * @param attributes conditions the value was produced under
   */
  default void put(String key, float[] embedding, V value, Map<String, String> attributes) {
    Objects.requireNonNull(attributes, "attributes");
    if (!attributes.isEmpty()) {
      throw new UnsupportedOperationException(
          getClass().getName() + " cannot store entry attributes");
    }
    put(key, embedding, value);
  }

  /**
   * Exact insert or replace for a batch of entries. Implementations commit the batch atomically.
   */
  void putAll(Collection<Entry<V>> entries);

  /**
   * Nearest-neighbour lookup. Returns the value whose embedding scores above {@link #threshold()}
   * against {@code queryEmbedding} under the configured similarity, or empty otherwise.
   *
   * @param queryEmbedding the lookup embedding
   * @return a cache hit, if any
   */
  Optional<Hit<V>> lookup(float[] queryEmbedding);

  /**
   * Finds the nearest cached entry whose attributes the filter accepts.
   *
   * <p>Filtering happens inside the search, not by testing the single nearest neighbour after the
   * fact. The nearest entry overall is frequently not the nearest entry the caller may use — a
   * verbatim repeat of the prompt at the wrong temperature sits closer than a paraphrase at the
   * right one — so post-filtering a top-1 result reports a miss where a usable entry exists.
   *
   * @param queryEmbedding embedding of the incoming query
   * @param filter conditions an entry must satisfy to be served
   * @return the closest acceptable entry within the threshold
   */
  default Optional<Hit<V>> lookup(float[] queryEmbedding, CacheFilter filter) {
    Objects.requireNonNull(filter, "filter");
    for (Hit<V> hit : lookupTopK(queryEmbedding, FILTER_CANDIDATES)) {
      if (filter.test(hit.attributes())) {
        return Optional.of(hit);
      }
    }
    return Optional.empty();
  }

  /**
   * Returns up to {@code k} nearest entries within the threshold, closest first.
   *
   * <p>The default implementation degrades to the single nearest entry, which makes {@link
   * #lookup(float[], CacheFilter)} correct but weak on an implementation that has not overridden
   * it: a filtered lookup then only ever considers one candidate.
   *
   * @param queryEmbedding embedding of the incoming query
   * @param k maximum number of entries to return
   * @return the nearest entries, closest first, possibly empty
   */
  default List<Hit<V>> lookupTopK(float[] queryEmbedding, int k) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be positive");
    }
    return lookup(queryEmbedding).map(List::of).orElseGet(List::of);
  }

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
   * @param key the exact key of the matched entry (the key it was {@link #put} under); lets callers
   *     tell <em>which</em> cached entry answered a near-duplicate lookup, not just its payload
   * @param value the stored payload
   * @param score similarity score between the lookup embedding and the matched entry
   * @param <V> payload type
   */
  record Hit<V>(String key, V value, double score, Map<String, String> attributes) {

    /** Validates and defensively copies the attributes. */
    public Hit {
      attributes = Map.copyOf(Objects.requireNonNull(attributes, "attributes"));
    }

    /**
     * Creates a hit carrying no attributes.
     *
     * @param key exact cache key
     * @param value cached payload
     * @param score similarity score
     */
    public Hit(String key, V value, double score) {
      this(key, value, score, Map.of());
    }
  }

  /**
   * A semantic-cache entry to insert.
   *
   * @param key exact cache key
   * @param embedding embedding associated with the cached value
   * @param value payload to cache
   * @param <V> payload type
   */
  record Entry<V>(String key, float[] embedding, V value, Map<String, String> attributes) {

    /** Validates and defensively copies the embedding and attributes. */
    public Entry {
      Objects.requireNonNull(key, "key");
      embedding = Objects.requireNonNull(embedding, "embedding").clone();
      attributes = Map.copyOf(Objects.requireNonNull(attributes, "attributes"));
    }

    /**
     * Creates an entry carrying no attributes.
     *
     * @param key exact cache key
     * @param embedding embedding associated with the cached value
     * @param value payload to cache
     */
    public Entry(String key, float[] embedding, V value) {
      this(key, embedding, value, Map.of());
    }

    @Override
    public float[] embedding() {
      return embedding.clone();
    }
  }
}
