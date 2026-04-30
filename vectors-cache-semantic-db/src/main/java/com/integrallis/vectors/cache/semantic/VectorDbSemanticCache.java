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
package com.integrallis.vectors.cache.semantic;

import com.integrallis.vectors.cache.CacheAdmissionPolicy;
import com.integrallis.vectors.cache.CacheStats;
import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.LongAdder;

/**
 * {@link SemanticCache} backed by a java-vectors {@link VectorCollection}.
 *
 * <p>Each cache entry is stored as a {@link Document} whose id is the caller-supplied key and whose
 * metadata carries the serialized payload under the {@code cached.payload} key (via the supplied
 * {@link PayloadCodec}). Lookups run a top-1 k-NN search; a hit is returned iff the score clears
 * the configured {@link #threshold()}.
 *
 * <p>The threshold must be interpreted in the same similarity space as the backing collection:
 * cosine in {@code [-1, 1]}, Euclidean in {@code [0, ∞)} (lower is better — a <i>higher</i> score
 * floor is enforced by inverting the check; use {@link Builder#thresholdRaw(double, boolean)} to
 * control semantics explicitly). The default is cosine with threshold {@code 0.92}.
 *
 * @param <V> payload type
 */
public final class VectorDbSemanticCache<V> implements SemanticCache<V> {

  /** Metadata field under which the serialized payload is stored on each cached Document. */
  public static final String PAYLOAD_FIELD = "cached.payload";

  private final VectorCollection collection;
  private final PayloadCodec<V> codec;
  private final double threshold;
  private final boolean higherIsBetter;
  private final boolean closeCollection;
  private final CacheAdmissionPolicy<V> admissionPolicy;

  private final LongAdder hits = new LongAdder();
  private final LongAdder misses = new LongAdder();
  private final LongAdder rejections = new LongAdder();

  VectorDbSemanticCache(
      VectorCollection collection,
      PayloadCodec<V> codec,
      double threshold,
      boolean higherIsBetter,
      boolean closeCollection,
      CacheAdmissionPolicy<V> admissionPolicy) {
    this.collection = Objects.requireNonNull(collection, "collection");
    this.codec = Objects.requireNonNull(codec, "codec");
    this.threshold = threshold;
    this.higherIsBetter = higherIsBetter;
    this.closeCollection = closeCollection;
    this.admissionPolicy = Objects.requireNonNull(admissionPolicy, "admissionPolicy");
  }

  /** Fluent builder. See {@link Builder}. */
  public static <V> Builder<V> builder(VectorCollection collection, PayloadCodec<V> codec) {
    return new Builder<>(collection, codec);
  }

  @Override
  public Optional<V> get(String key) {
    Document doc = collection.get(Objects.requireNonNull(key, "key"));
    return Optional.ofNullable(doc).map(this::decodePayload);
  }

  @Override
  public void put(String key, float[] embedding, V value) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(embedding, "embedding");
    Objects.requireNonNull(value, "value");
    if (!admissionPolicy.test(value)) {
      rejections.increment();
      return;
    }
    Map<String, MetadataValue> md =
        Map.of(PAYLOAD_FIELD, new MetadataValue.Str(codec.encode(value)));
    Document doc = new Document(key, embedding, null, md);
    collection.upsert(doc);
    collection.commit();
  }

  @Override
  public Optional<Hit<V>> lookup(float[] queryEmbedding) {
    Objects.requireNonNull(queryEmbedding, "queryEmbedding");
    SearchRequest req =
        SearchRequest.builder(queryEmbedding, 1)
            .includeVector(false)
            .includeText(false)
            .includeMetadata(true)
            .build();
    SearchResult result = collection.search(req);
    if (result.hits().isEmpty()) {
      misses.increment();
      return Optional.empty();
    }
    SearchResult.Hit top = result.hits().get(0);
    boolean pass = higherIsBetter ? top.score() >= threshold : top.score() <= threshold;
    if (!pass) {
      misses.increment();
      return Optional.empty();
    }
    hits.increment();
    return Optional.of(new Hit<>(decodePayload(top.document()), top.score()));
  }

  @Override
  public void invalidate(String key) {
    if (collection.delete(Objects.requireNonNull(key, "key"))) {
      collection.commit();
    }
  }

  @Override
  public void invalidateAll() {
    for (Document d : collection.documents()) {
      collection.delete(d.id());
    }
    collection.commit();
  }

  @Override
  public CacheStats stats() {
    return new CacheStats(hits.sum(), misses.sum(), 0L, rejections.sum(), collection.size());
  }

  @Override
  public double threshold() {
    return threshold;
  }

  @Override
  public CacheAdmissionPolicy<V> admissionPolicy() {
    return admissionPolicy;
  }

  @Override
  public void close() {
    if (closeCollection) {
      collection.close();
    }
  }

  private V decodePayload(Document doc) {
    MetadataValue raw = doc.metadata().get(PAYLOAD_FIELD);
    if (raw == null) {
      throw new IllegalStateException(
          "cached document '" + doc.id() + "' has no metadata field '" + PAYLOAD_FIELD + "'");
    }
    if (!(raw instanceof MetadataValue.Str str)) {
      throw new IllegalStateException(
          "cached document '"
              + doc.id()
              + "' payload field '"
              + PAYLOAD_FIELD
              + "' is "
              + raw.getClass().getSimpleName()
              + ", expected Str");
    }
    return codec.decode(str.value());
  }

  /** Fluent builder for {@link VectorDbSemanticCache}. */
  public static final class Builder<V> {
    private final VectorCollection collection;
    private final PayloadCodec<V> codec;
    private double threshold = 0.92;
    private boolean higherIsBetter = true;
    private boolean closeCollection = false;
    private CacheAdmissionPolicy<V> admissionPolicy = CacheAdmissionPolicy.allowAll();

    Builder(VectorCollection collection, PayloadCodec<V> codec) {
      this.collection = Objects.requireNonNull(collection, "collection");
      this.codec = Objects.requireNonNull(codec, "codec");
    }

    /** Cosine/dot-product threshold (higher is better). Default 0.92. */
    public Builder<V> threshold(double threshold) {
      this.threshold = threshold;
      this.higherIsBetter = true;
      return this;
    }

    /**
     * Raw threshold with explicit comparison direction. Use this for Euclidean collections where
     * {@code higherIsBetter} should be {@code false}.
     */
    public Builder<V> thresholdRaw(double threshold, boolean higherIsBetter) {
      this.threshold = threshold;
      this.higherIsBetter = higherIsBetter;
      return this;
    }

    /** If true, {@link VectorDbSemanticCache#close()} closes the backing collection. */
    public Builder<V> closeCollectionOnClose(boolean close) {
      this.closeCollection = close;
      return this;
    }

    /**
     * Sets the admission policy that gates {@link #put} calls. Default is {@link
     * CacheAdmissionPolicy#allowAll()}.
     */
    public Builder<V> admissionPolicy(CacheAdmissionPolicy<V> policy) {
      this.admissionPolicy = Objects.requireNonNull(policy, "admissionPolicy");
      return this;
    }

    public VectorDbSemanticCache<V> build() {
      return new VectorDbSemanticCache<>(
          collection, codec, threshold, higherIsBetter, closeCollection, admissionPolicy);
    }
  }
}
