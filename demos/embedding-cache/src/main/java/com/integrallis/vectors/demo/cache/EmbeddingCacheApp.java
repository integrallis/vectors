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
package com.integrallis.vectors.demo.cache;

import com.integrallis.vectors.cache.CacheStats;
import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import com.integrallis.vectors.cache.langchain4j.CachingEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Embedding-cache demo using {@link CachingEmbeddingModel}.
 *
 * <p>Wraps a LangChain4j {@link EmbeddingModel} with {@link CachingEmbeddingModel}, backed by a
 * {@link CaffeineVectorCache}. The decorator is a drop-in replacement — any framework-visible call
 * site that talks to {@code EmbeddingModel} automatically benefits from the cache without code
 * changes elsewhere.
 *
 * <p>This demo shows:
 *
 * <ol>
 *   <li>Single-text and batch embed paths both consult the cache.
 *   <li>All batch misses are coalesced into a single delegate {@code embedAll} round-trip.
 *   <li>Re-embedding already-seen texts hits the cache without invoking the delegate.
 * </ol>
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:embedding-cache:run
 * </pre>
 */
public final class EmbeddingCacheApp {

  private static final int DIMENSION = 128;

  private EmbeddingCacheApp() {}

  public static void main(String[] args) {
    AtomicInteger delegateCalls = new AtomicInteger();
    EmbeddingModel realModel = new CountingEmbeddingModel(DIMENSION, delegateCalls);

    try (VectorCache<String, float[]> cache =
        CaffeineVectorCache.<String, float[]>builder()
            .maximumSize(10_000L)
            .expireAfterWrite(Duration.ofHours(1))
            .build()) {

      EmbeddingModel cached = new CachingEmbeddingModel(realModel, cache);

      List<String> phase1 =
          List.of(
              "What is HNSW?",
              "Explain product quantization.",
              "How does mmap persistence work?",
              "What is HNSW?", // duplicate within this batch
              "Describe scalar quantization.",
              "Explain product quantization."); // duplicate
      System.out.println("--- phase 1: batch embed with duplicates ---");
      embedBatch(cached, phase1);
      printStats("after phase 1", cache.stats(), delegateCalls.get());

      List<String> phase2 =
          List.of(
              "What is HNSW?", // hit
              "Explain product quantization.", // hit
              "What is ANN?"); // miss
      System.out.println("\n--- phase 2: batch embed that is mostly hits ---");
      embedBatch(cached, phase2);
      printStats("after phase 2", cache.stats(), delegateCalls.get());

      System.out.println("\n--- phase 3: single-text embed on a hot key ---");
      Response<Embedding> r = cached.embed("What is HNSW?");
      System.out.printf(
          "  embed(\"What is HNSW?\") -> dim=%d, first=%f%n",
          r.content().vector().length, r.content().vector()[0]);
      printStats("after phase 3", cache.stats(), delegateCalls.get());
    }
  }

  private static void embedBatch(EmbeddingModel model, List<String> texts) {
    List<TextSegment> segments = texts.stream().map(TextSegment::from).toList();
    Response<List<Embedding>> response = model.embedAll(segments);
    for (int i = 0; i < texts.size(); i++) {
      float first = response.content().get(i).vector()[0];
      System.out.printf("  %-40s  vec[0]=%.4f%n", texts.get(i), first);
    }
  }

  private static void printStats(String label, CacheStats stats, int delegateCalls) {
    System.out.printf(
        "%s: hits=%d, misses=%d, entries=%d, hitRate=%.1f%%, delegate embed calls=%d%n",
        label, stats.hits(), stats.misses(), stats.size(), 100.0 * stats.hitRate(), delegateCalls);
  }

  /**
   * Zero-dependency {@link EmbeddingModel} that counts its invocations. Bag-of-character-trigrams
   * vector, L2-normalized — similar texts produce similar vectors without requiring a model
   * download. Not for production.
   */
  private static final class CountingEmbeddingModel implements EmbeddingModel {
    private final int dimension;
    private final AtomicInteger singleCalls;

    CountingEmbeddingModel(int dimension, AtomicInteger singleCalls) {
      this.dimension = dimension;
      this.singleCalls = singleCalls;
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public Response<Embedding> embed(String text) {
      singleCalls.incrementAndGet();
      return Response.from(Embedding.from(trigramVector(Objects.requireNonNull(text))));
    }

    @Override
    public Response<Embedding> embed(TextSegment segment) {
      return embed(segment.text());
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
      List<Embedding> out = new ArrayList<>(segments.size());
      for (TextSegment s : segments) {
        singleCalls.incrementAndGet();
        out.add(Embedding.from(trigramVector(s.text())));
      }
      return Response.from(out);
    }

    private float[] trigramVector(String text) {
      float[] out = new float[dimension];
      String normalized = text.toLowerCase();
      if (normalized.length() < 3) normalized = "  " + normalized + "  ";
      for (int i = 0; i <= normalized.length() - 3; i++) {
        int h =
            31 * (31 * normalized.charAt(i) + normalized.charAt(i + 1)) + normalized.charAt(i + 2);
        out[Math.floorMod(h, dimension)] += 1.0f;
      }
      float norm = 0f;
      for (float v : out) norm += v * v;
      norm = (float) Math.sqrt(norm);
      if (norm > 0f) for (int i = 0; i < dimension; i++) out[i] /= norm;
      return out;
    }
  }
}
