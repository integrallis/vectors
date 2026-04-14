package com.integrallis.vectors.storage.backend;

import java.io.IOException;
import java.util.List;

/**
 * Abstraction over a key-value object store. Implementations range from an in-heap map ({@link
 * HeapStorageBackend}) to a local filesystem ({@link LocalFileStorageBackend}) to a remote
 * S3-compatible store (phase 3, {@code S3StorageBackend}).
 *
 * <p>Keys are arbitrary non-null, non-empty strings. Values are raw byte arrays. All operations are
 * synchronous and may throw {@link IOException} on persistent errors.
 *
 * <p>Conditional put implements optimistic compare-and-swap semantics: the caller provides an
 * {@code expectedEtag} (null = "key must not exist"); the backend only writes if the current etag
 * matches. The etag is an opaque content-based identifier produced by the backend.
 */
public interface StorageBackend {

  /**
   * Unconditionally writes {@code value} under {@code key}, overwriting any existing value.
   *
   * @throws IOException on storage failure
   */
  void put(String key, byte[] value) throws IOException;

  /**
   * Returns the value stored under {@code key}, or {@code null} if the key does not exist.
   *
   * @throws IOException on storage failure
   */
  byte[] get(String key) throws IOException;

  /**
   * Returns all keys whose string representation starts with {@code prefix}, in undefined order.
   *
   * @throws IOException on storage failure
   */
  List<String> list(String prefix) throws IOException;

  /**
   * Removes {@code key} and its value. No-op if the key does not exist.
   *
   * @throws IOException on storage failure
   */
  void delete(String key) throws IOException;

  /**
   * Atomically writes {@code value} under {@code key} only if the current etag equals {@code
   * expectedEtag}.
   *
   * <ul>
   *   <li>Pass {@code expectedEtag = null} to assert the key does not yet exist.
   *   <li>On success, returns {@code {succeeded=true, newEtag=<content-based-tag>}}.
   *   <li>On failure (etag mismatch), returns {@code {succeeded=false, newEtag=null}}; existing
   *       value is unchanged.
   * </ul>
   *
   * @throws IOException on storage failure
   */
  ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag)
      throws IOException;

  /** Result of a {@link #conditionalPut} operation. */
  record ConditionalPutResult(boolean succeeded, String newEtag) {}
}
