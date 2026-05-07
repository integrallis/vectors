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
   * Returns exactly {@code length} bytes from the value stored under {@code key}, starting at byte
   * {@code offset}. Used to fetch a single vector ordinal or a sub-slice of a WAL segment without
   * downloading the full object — the S3 implementation issues an HTTP {@code Range} header so only
   * the requested bytes traverse the network.
   *
   * <ul>
   *   <li>Returns {@code null} if {@code key} does not exist (mirrors {@link #get}).
   *   <li>Returns an empty array when {@code length == 0}.
   *   <li>Throws {@link IndexOutOfBoundsException} if {@code offset < 0}, {@code length < 0}, or
   *       {@code offset + length} exceeds the stored value size.
   * </ul>
   *
   * @throws IOException on storage failure
   * @throws IndexOutOfBoundsException if the requested range is invalid for the stored value
   */
  byte[] getRange(String key, long offset, int length) throws IOException;

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
