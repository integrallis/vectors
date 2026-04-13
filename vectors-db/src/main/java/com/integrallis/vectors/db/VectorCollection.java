package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import java.util.Collection;

/**
 * Public facade for an embedded vector database collection.
 *
 * <p>Supports FLAT, HNSW, and VAMANA backends in both in-memory and mmap-persistent modes. Deletion
 * uses tombstone semantics: {@code delete}/{@code deleteWhere} stage tombstones that take effect on
 * the next {@code commit}. {@code compact()} rebuilds with dense ordinals, reclaiming space.
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

  /** No-op in Step 3 (in-memory only). Step 4 adds fsync semantics over the persistence layer. */
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

  /** Returns the document with the given id, or {@code null} if unknown. */
  Document get(String id);

  /** Returns {@code true} if the id is present in the live generation. */
  boolean contains(String id);

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
