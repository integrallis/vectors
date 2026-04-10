package com.integrallis.vectors.db;

import com.integrallis.vectors.db.filter.Filter;
import java.util.Collection;

/**
 * Public facade for an embedded vector database collection.
 *
 * <p>Step 3 subset: in-memory only, flat-scan reference backend, no persistence, no filter
 * execution beyond {@link com.integrallis.vectors.db.filter.Filter.All}. {@code upsert}, {@code
 * delete}, {@code deleteWhere}, and {@code compact} throw {@link UnsupportedOperationException}.
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
   * Upserts a document (insert or replace). Deferred to Step 6.
   *
   * @throws UnsupportedOperationException always
   */
  default void upsert(Document doc) {
    throw new UnsupportedOperationException("upsert deferred to Step 6");
  }

  /**
   * Deletes a document by id. Deferred to Step 6.
   *
   * @throws UnsupportedOperationException always
   */
  default void delete(String id) {
    throw new UnsupportedOperationException("delete deferred to Step 6");
  }

  /**
   * Deletes documents matching a filter. Deferred to Step 5.
   *
   * @throws UnsupportedOperationException always
   */
  default void deleteWhere(Filter filter) {
    throw new UnsupportedOperationException("deleteWhere deferred to Step 5");
  }

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
   * Compacts tombstoned documents out of the current generation. Deferred to Step 6.
   *
   * @throws UnsupportedOperationException always
   */
  default void compact() {
    throw new UnsupportedOperationException("compact deferred to Step 6");
  }

  /** Searches the currently-committed generation. */
  SearchResult search(SearchRequest request);

  /** Returns the document with the given id, or {@code null} if unknown. */
  Document get(String id);

  /** Returns {@code true} if the id is present in the live generation. */
  boolean contains(String id);

  /** Number of documents in the currently-committed generation. */
  int size();

  /**
   * Number of vectors physically stored across live and staged buffers (including tombstones in
   * later steps). Equal to {@link #size()} in Step 3.
   */
  int physicalSize();

  /** Returns the collection's config. */
  VectorCollectionConfig config();

  @Override
  void close();
}
