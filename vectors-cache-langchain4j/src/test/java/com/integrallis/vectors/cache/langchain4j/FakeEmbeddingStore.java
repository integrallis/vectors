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

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/** Minimal in-memory EmbeddingStore with cosine scoring for tests. */
final class FakeEmbeddingStore<T> implements EmbeddingStore<T> {

  private record Entry<T>(String id, Embedding embedding, T payload) {}

  private final ConcurrentHashMap<String, Entry<T>> entries = new ConcurrentHashMap<>();
  final AtomicInteger searchCalls = new AtomicInteger();

  @Override
  public String add(Embedding embedding) {
    return add(embedding, null);
  }

  @Override
  public void add(String id, Embedding embedding) {
    entries.put(id, new Entry<>(id, embedding, null));
  }

  @Override
  public String add(Embedding embedding, T embedded) {
    String id = UUID.randomUUID().toString();
    entries.put(id, new Entry<>(id, embedding, embedded));
    return id;
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings) {
    List<String> ids = new ArrayList<>(embeddings.size());
    for (Embedding e : embeddings) {
      ids.add(add(e));
    }
    return ids;
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings, List<T> embedded) {
    List<String> ids = new ArrayList<>(embeddings.size());
    for (int i = 0; i < embeddings.size(); i++) {
      ids.add(add(embeddings.get(i), embedded.get(i)));
    }
    return ids;
  }

  @Override
  public void addAll(List<String> ids, List<Embedding> embeddings, List<T> embedded) {
    for (int i = 0; i < ids.size(); i++) {
      entries.put(ids.get(i), new Entry<>(ids.get(i), embeddings.get(i), embedded.get(i)));
    }
  }

  @Override
  public void remove(String id) {
    entries.remove(id);
  }

  @Override
  public void removeAll(Collection<String> ids) {
    ids.forEach(entries::remove);
  }

  @Override
  public void removeAll(Filter filter) {
    entries.values().removeIf(entry -> filter.test(entry.payload()));
  }

  @Override
  public void removeAll() {
    entries.clear();
  }

  @Override
  public EmbeddingSearchResult<T> search(EmbeddingSearchRequest request) {
    searchCalls.incrementAndGet();
    float[] q = request.queryEmbedding().vector();
    int maxResults = request.maxResults();
    double minScore = request.minScore();

    List<EmbeddingMatch<T>> matches = new ArrayList<>();
    for (Entry<T> e : entries.values()) {
      double score = cosine(q, e.embedding().vector());
      if (score >= minScore) {
        matches.add(new EmbeddingMatch<>(score, e.id(), e.embedding(), e.payload()));
      }
    }
    matches.sort((a, b) -> Double.compare(b.score(), a.score()));
    if (matches.size() > maxResults) {
      matches = new ArrayList<>(matches.subList(0, maxResults));
    }
    return new EmbeddingSearchResult<>(matches);
  }

  private static double cosine(float[] a, float[] b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    int n = Math.min(a.length, b.length);
    for (int i = 0; i < n; i++) {
      dot += (double) a[i] * b[i];
      na += (double) a[i] * a[i];
      nb += (double) b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) {
      return 0.0;
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
}
