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
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Caching decorator around a LangChain4j {@link EmbeddingStore}. Search results are cached in the
 * supplied {@link VectorCache} keyed by {@link EmbeddingSearchRequest} (which has a value-based
 * {@code equals}/{@code hashCode}). Every mutating operation invalidates the entire search cache to
 * avoid serving stale results.
 *
 * <p>Write-through semantics: {@code add}/{@code addAll}/{@code remove}/{@code removeAll} are
 * forwarded verbatim to the delegate, then the search cache is cleared. For workloads with a large
 * read-to-write ratio this wins even with full invalidation; workloads that mutate after every read
 * should not use a cache in front of the store.
 *
 * @param <T> the embedded payload type (typically {@code TextSegment})
 */
public class CachingEmbeddingStore<T> implements EmbeddingStore<T> {

  private final EmbeddingStore<T> delegate;
  private final VectorCache<EmbeddingSearchRequest, EmbeddingSearchResult<T>> searchCache;
  private final ReadWriteLock lock = new ReentrantReadWriteLock();

  /**
   * @param delegate non-null underlying store
   * @param searchCache non-null cache keyed by search request
   */
  public CachingEmbeddingStore(
      EmbeddingStore<T> delegate,
      VectorCache<EmbeddingSearchRequest, EmbeddingSearchResult<T>> searchCache) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.searchCache = Objects.requireNonNull(searchCache, "searchCache");
  }

  /**
   * @return the wrapped delegate
   */
  public final EmbeddingStore<T> delegate() {
    return delegate;
  }

  @Override
  public String add(Embedding embedding) {
    lock.writeLock().lock();
    try {
      String id = delegate.add(embedding);
      invalidate();
      return id;
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void add(String id, Embedding embedding) {
    lock.writeLock().lock();
    try {
      delegate.add(id, embedding);
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public String add(Embedding embedding, T embedded) {
    lock.writeLock().lock();
    try {
      String id = delegate.add(embedding, embedded);
      invalidate();
      return id;
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings) {
    lock.writeLock().lock();
    try {
      List<String> ids = delegate.addAll(embeddings);
      invalidate();
      return ids;
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public List<String> addAll(List<Embedding> embeddings, List<T> embedded) {
    lock.writeLock().lock();
    try {
      List<String> ids = delegate.addAll(embeddings, embedded);
      invalidate();
      return ids;
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void addAll(List<String> ids, List<Embedding> embeddings, List<T> embedded) {
    lock.writeLock().lock();
    try {
      delegate.addAll(ids, embeddings, embedded);
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void remove(String id) {
    lock.writeLock().lock();
    try {
      delegate.remove(id);
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void removeAll(Collection<String> ids) {
    lock.writeLock().lock();
    try {
      delegate.removeAll(ids);
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void removeAll(Filter filter) {
    lock.writeLock().lock();
    try {
      delegate.removeAll(filter);
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void removeAll() {
    lock.writeLock().lock();
    try {
      delegate.removeAll();
      invalidate();
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public EmbeddingSearchResult<T> search(EmbeddingSearchRequest request) {
    Objects.requireNonNull(request, "request");
    lock.readLock().lock();
    try {
      return searchCache.getOrCompute(request, delegate::search);
    } finally {
      lock.readLock().unlock();
    }
  }

  private void invalidate() {
    searchCache.invalidateAll();
  }
}
