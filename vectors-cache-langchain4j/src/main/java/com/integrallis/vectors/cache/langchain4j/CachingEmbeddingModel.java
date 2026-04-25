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

import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Caching decorator around a LangChain4j {@link EmbeddingModel}. All text-based embedding paths are
 * served from the supplied {@link VectorCache}; misses are forwarded to the delegate and the result
 * is cached under the normalized key.
 *
 * <p>The single-text {@link #embed(String)} and {@link #embed(TextSegment)} paths share keys — only
 * the raw text is hashed, segment metadata is ignored. {@link #embedAll(List)} coalesces cache
 * misses into a single delegate round-trip.
 */
public class CachingEmbeddingModel implements EmbeddingModel {

  private static final Logger LOG = Logger.getLogger(CachingEmbeddingModel.class.getName());

  private final EmbeddingModel delegate;
  private final VectorCache<String, float[]> cache;
  private final UnaryOperator<String> keyFn;
  private final int dimension;

  /**
   * @see #CachingEmbeddingModel(EmbeddingModel, VectorCache, UnaryOperator)
   */
  public CachingEmbeddingModel(EmbeddingModel delegate, VectorCache<String, float[]> cache) {
    this(delegate, cache, UnaryOperator.identity());
  }

  /**
   * @param delegate non-null underlying model
   * @param cache non-null key/value cache keyed by normalized text
   * @param keyFn text-normalization function applied before cache lookup and store
   */
  public CachingEmbeddingModel(
      EmbeddingModel delegate, VectorCache<String, float[]> cache, UnaryOperator<String> keyFn) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.cache = Objects.requireNonNull(cache, "cache");
    this.keyFn = Objects.requireNonNull(keyFn, "keyFn");
    this.dimension = detectDimension(delegate);
  }

  /**
   * @return the wrapped delegate
   */
  public final EmbeddingModel delegate() {
    return delegate;
  }

  @Override
  public Response<Embedding> embed(String text) {
    Objects.requireNonNull(text, "text");
    float[] v = cache.getOrCompute(keyFn.apply(text), k -> delegate.embed(text).content().vector());
    return Response.from(Embedding.from(v.clone()));
  }

  @Override
  public Response<Embedding> embed(TextSegment segment) {
    Objects.requireNonNull(segment, "segment");
    return embed(segment.text());
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    Objects.requireNonNull(segments, "segments");
    int n = segments.size();
    Embedding[] out = new Embedding[n];
    List<Integer> missIndexes = new ArrayList<>();
    List<TextSegment> missSegments = new ArrayList<>();
    List<String> missKeys = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      TextSegment seg = Objects.requireNonNull(segments.get(i), "segments[i]");
      String key = keyFn.apply(seg.text());
      var hit = cache.get(key);
      if (hit.isPresent()) {
        out[i] = Embedding.from(hit.get().clone());
      } else {
        missIndexes.add(i);
        missSegments.add(seg);
        missKeys.add(key);
      }
    }
    if (!missSegments.isEmpty()) {
      List<Embedding> fresh = delegate.embedAll(missSegments).content();
      if (fresh.size() != missSegments.size()) {
        throw new IllegalStateException(
            "delegate returned "
                + fresh.size()
                + " embeddings for "
                + missSegments.size()
                + " inputs");
      }
      for (int j = 0; j < fresh.size(); j++) {
        int i = missIndexes.get(j);
        Embedding e = fresh.get(j);
        out[i] = e;
        cache.put(missKeys.get(j), e.vector());
      }
    }
    return Response.from(List.of(out));
  }

  @Override
  public int dimension() {
    return dimension > 0 ? dimension : delegate.dimension();
  }

  private static int detectDimension(EmbeddingModel model) {
    try {
      return model.dimension();
    } catch (RuntimeException e) {
      LOG.log(
          Level.WARNING,
          "failed to detect dimension from delegate; dimension() will delegate on each call",
          e);
      return -1;
    }
  }
}
