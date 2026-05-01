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
package com.integrallis.vectors.hybrid.text;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Extension point for full-text search, metadata storage, and blob storage.
 *
 * <p>Implementations are discovered via {@link java.util.ServiceLoader}. The vectors-server creates
 * one instance per collection when a provider is available on the classpath.
 */
public interface TextIndexSpi extends AutoCloseable {

  /** Index a batch of documents for full-text search. */
  void index(List<TextDocument> documents);

  /**
   * Full-text search returning ranked document IDs.
   *
   * @param query the search query text
   * @param k maximum number of results
   * @return ranked results (descending by relevance)
   */
  TextSearchOutcome search(String query, int k);

  /**
   * Retrieve stored content by document ID.
   *
   * @param id the document identifier
   * @return the stored content, or empty if not found
   */
  Optional<StoredContent> get(String id);

  /**
   * Retrieve a blob (e.g. image bytes) by document ID.
   *
   * @param id the document identifier
   * @return the blob bytes, or empty if not found
   */
  Optional<byte[]> getBlob(String id);

  /** Remove a document from the text index. */
  void remove(String id);

  /** Clear all data from this text index. */
  void clear();

  /** Returns the number of indexed documents. */
  int size();

  @Override
  void close();

  /**
   * Permanently destroys this text index, closing the connection and deleting all on-disk files.
   * After calling this method, the index cannot be reopened. The default implementation delegates
   * to {@link #close()}.
   */
  default void drop() {
    close();
  }

  /**
   * A document to index.
   *
   * @param id the document identifier
   * @param text the text content for full-text search
   * @param metadata key-value metadata
   * @param blob optional binary data (e.g. image bytes), may be null
   */
  record TextDocument(String id, String text, Map<String, String> metadata, byte[] blob) {

    public TextDocument {
      if (id == null || id.isBlank()) {
        throw new IllegalArgumentException("id must not be null or blank");
      }
    }
  }

  /**
   * Stored content retrieved by ID.
   *
   * @param id the document identifier
   * @param text the stored text content
   * @param metadata key-value metadata
   */
  record StoredContent(String id, String text, Map<String, String> metadata) {}
}
