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
package com.integrallis.vectors.db;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.filter.Filter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Public facade for an embedded vector database collection.
 *
 * <p>Supports FLAT, HNSW, VAMANA, IVF_FLAT, and IVF_PQ backends in both in-memory and
 * mmap-persistent modes. Deletion uses tombstone semantics: {@code delete}/{@code deleteWhere}
 * stage tombstones that take effect on the next {@code commit}. {@code compact()} rebuilds with
 * dense ordinals, reclaiming space.
 *
 * <p>Lifecycle:
 *
 * <ol>
 *   <li>Configure and create a collection via {@link #builder()}.
 *   <li>{@link #add(Document)} or {@link #addAll(Collection)} documents. They accumulate in a
 *       staging buffer and are not visible to {@link #search(SearchRequest)} until commit.
 *   <li>{@link #commit()} atomically installs a new backend containing the previously live
 *       documents plus the newly staged ones. Commits are published via a volatile snapshot so
 *       concurrent searches are never blocked.
 *   <li>If {@code autoCommitThreshold} has been set on the builder, {@code add}/{@code addAll} will
 *       auto-commit before returning as soon as the staging buffer reaches that size. By default
 *       the threshold is {@link Integer#MAX_VALUE}, so the caller must drive commits explicitly.
 *   <li>{@link #search(SearchRequest)} returns hits from the last committed generation.
 *   <li>{@link #close()} releases resources.
 * </ol>
 */
public interface VectorCollection extends AutoCloseable {

  /** Creates a new builder for an in-memory collection. */
  static VectorCollectionBuilder builder() {
    return new VectorCollectionBuilder();
  }

  /**
   * Stages a document for inclusion in the next committed generation.
   *
   * @throws IllegalArgumentException if the vector dimension doesn't match the collection config or
   *     the id is already known in the live or staged set
   */
  void add(Document doc);

  /** Stages a batch of documents. Equivalent to calling {@link #add(Document)} for each one. */
  void addAll(Collection<Document> docs);

  /**
   * Upserts a document (insert or replace). If a document with the same id already exists in the
   * live generation or the staging buffer, it is replaced. If the id is unknown, the document is
   * staged as a new add.
   *
   * @throws IllegalArgumentException if the vector dimension doesn't match the collection config
   */
  void upsert(Document doc);

  /**
   * Stages a tombstone for the document with the given id. The document is excluded from search,
   * get, and contains after the next commit. If the id refers to a staged (not-yet-committed) add,
   * the staged document is removed immediately.
   *
   * @return {@code true} if the document was found (in live generation or staging) and marked for
   *     deletion; {@code false} if the id is unknown or already deleted
   */
  boolean delete(String id);

  /**
   * Stages tombstones for all documents whose metadata matches the given filter.
   *
   * @return the number of documents matched and marked for deletion
   */
  int deleteWhere(Filter filter);

  /**
   * Atomically installs a new generation containing the currently-live documents plus any staged
   * additions since the previous commit. After a successful commit, subsequent {@link
   * #search(SearchRequest)} calls see the new generation and the staging buffer is empty.
   */
  void commit();

  /**
   * Reloads the newest committed generation from the storage root, if a newer one has appeared
   * since this collection last loaded — the read-replica refresh point (P3.1). A follower whose
   * storage root is being populated by a {@link GenerationShippingSubscriber} (directly, or via
   * {@link GenerationSync#pull}) calls this to start serving newly-shipped generations.
   *
   * <p>Safe to call repeatedly (e.g. on a poll loop). A generation that is only partially present
   * is skipped until complete. Returns {@code false} for in-memory collections (nothing to reload).
   *
   * @return {@code true} if a newer generation was loaded and is now being served
   */
  default boolean refresh() {
    return false;
  }

  /**
   * Returns the current committed generation number, or {@code -1} if the collection does not track
   * generations. For a persistent collection this advances on every commit and is the value carried
   * by a {@link CommitToken}.
   */
  default long generationNumber() {
    return -1L;
  }

  /** Flushes durable state for persistent collections; in-memory collections have nothing to do. */
  default void flush() {
    // in-memory: nothing to flush
  }

  /**
   * Compacts tombstoned documents out of the current generation, rebuilding with dense ordinals.
   * After compact, {@link #physicalSize()} equals {@link #size()}. If there are no tombstones this
   * is a no-op. Any pending staged work is committed first.
   */
  void compact();

  /** Searches the currently-committed generation. */
  SearchResult search(SearchRequest request);

  /**
   * Searches the currently-committed generation for multiple queries in parallel.
   *
   * <p>Each query in {@code requests} is dispatched as an independent task on a virtual-thread
   * executor. This method blocks while collecting the results in request order. If a query fails,
   * its cause is wrapped in {@link RuntimeException}; closing the executor waits for already
   * submitted tasks to terminate.
   *
   * <p>The result list has the same size as {@code requests} and preserves request order: {@code
   * results.get(i)} corresponds to {@code requests.get(i)}.
   *
   * @param requests non-null, non-empty list of search requests
   * @return list of results in the same order as {@code requests}
   * @throws IllegalArgumentException if {@code requests} is null or empty
   */
  default List<SearchResult> searchBatch(List<SearchRequest> requests) {
    if (requests == null || requests.isEmpty()) {
      throw new IllegalArgumentException("requests must not be null or empty");
    }
    if (requests.size() == 1) {
      return List.of(search(requests.get(0)));
    }
    // One virtual thread per query: all dispatched simultaneously, joined in order.
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<SearchResult>> futures = new ArrayList<>(requests.size());
      for (SearchRequest req : requests) {
        futures.add(executor.submit(() -> search(req)));
      }
      List<SearchResult> out = new ArrayList<>(futures.size());
      for (Future<SearchResult> f : futures) {
        try {
          out.add(f.get());
        } catch (ExecutionException e) {
          throw new RuntimeException("searchBatch query failed", e.getCause());
        }
      }
      return List.copyOf(out);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("searchBatch interrupted", e);
    }
  }

  /** Returns the document with the given id, or {@code null} if unknown. */
  Document get(String id);

  /** Returns {@code true} if the id is present in the live generation. */
  boolean contains(String id);

  /**
   * Returns an immutable snapshot of all live (non-deleted) documents in the currently-committed
   * generation, with vectors fully populated (hydrated from mmap if needed for persistent
   * collections).
   *
   * <p>Useful for bulk export via {@link com.integrallis.vectors.db.arrow.ArrowIpcExporter} and
   * other serialization paths that need to iterate the full corpus.
   *
   * @return unmodifiable list of live documents; never {@code null}
   */
  List<Document> documents();

  /** Number of documents in the currently-committed generation. */
  int size();

  /**
   * Number of vectors physically allocated in the current committed generation, including ordinals
   * that have been tombstoned by {@link #delete}/{@link #deleteWhere} but not yet reclaimed by
   * {@link #compact()}. After {@code compact()} completes, equals {@link #size()}.
   */
  int physicalSize();

  /** Returns the collection's config. */
  VectorCollectionConfig config();

  @Override
  void close();
}
