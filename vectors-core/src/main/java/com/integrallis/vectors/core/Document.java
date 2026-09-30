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
package com.integrallis.vectors.core;

import java.util.Map;
import java.util.Objects;

/**
 * A document carrying an external string id, a float vector, optional text, and typed metadata.
 *
 * <p>The record's canonical constructor copies the metadata map defensively and rejects {@code
 * null} for {@code id}. The {@code vector} is mandatory at insertion time but may be {@code null}
 * in search-result projections when {@code SearchRequest.includeVector} is false. {@code text} and
 * {@code metadata} may also be null — a null metadata map is normalised to an empty map.
 *
 * <p><b>Vector ownership.</b> The {@code vector} array on a Document passed to a collection is
 * defensively cloned at the staging boundary, so the caller may safely reuse and mutate their own
 * buffer afterwards (e.g., to batch-insert from a reusable float[]). Conversely, the {@code vector}
 * array on a Document <i>returned</i> by a collection (via {@code search} projections or {@code
 * get}) references the collection's internally-held storage and <b>must not</b> be mutated — doing
 * so corrupts the stored vector and subsequent search results. Treat returned vector arrays as
 * immutable.
 *
 * @param id external identifier (must not be null)
 * @param vector embedding (required on insertion; may be null in projections; stored by reference,
 *     not copied)
 * @param text optional raw text (may be null)
 *     <p><b>Provenance.</b> {@code contentHash} is never absent. It is the SHA-256 of the input the
 *     vector was produced from, and it is what makes staleness decidable: a document needs
 *     re-embedding exactly when the hash of its current content stops matching. Storing the source
 *     is optional — content may already live in a system of record, and for an image or an audio
 *     clip there is no text to keep — but the hash is not, because a collection that cannot tell
 *     which of its vectors are stale cannot be maintained. That is the failure every
 *     bring-your-own-vectors store leaves to convention, and conventions are forgotten.
 *     <p>When {@code text} is present the hash is <b>derived from it</b>, so existing callers gain
 *     provenance without changing. Where {@code text} is null the hash must be supplied — see
 *     {@link #withHash} — which is also the only case where re-embedding from the collection alone
 *     is impossible.
 *     <p><b>Hash the embedded input.</b> Where a recipe applies a role prefix, hash {@link
 *     EmbeddingRecipe#documentInput} rather than the bare text, or the hash answers a question
 *     nobody asked. A single hash suffices because a collection's recipe is immutable: every vector
 *     in it came from the same generation, so a document is stale only if its own content moved.
 * @param metadata typed metadata map (null → empty)
 * @param contentHash SHA-256 of the embedded input, lowercase hex; derived from {@code text} when
 *     one is given, and required otherwise
 */
public record Document(
    String id,
    float[] vector,
    String text,
    Map<String, MetadataValue> metadata,
    String contentHash) {

  public Document {
    Objects.requireNonNull(id, "id must not be null");
    metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    if (contentHash == null) {
      // Derived when there is text to derive it from. Left null for a bare vector, which is a real
      // and common case -- compaction fixtures, generation shipping, and any bring-your-own-vectors
      // caller whose vector has no textual content at all. Requiring a hash of content that does
      // not
      // exist would refuse those, so the obligation belongs where content is actually claimed: a
      // collection configured with an EmbeddingRecipe enforces it on add. See
      // EmbeddingRecipe#requireProvenance.
      contentHash = text == null ? null : ContentHash.of(text);
    } else {
      contentHash = ContentHash.validated(contentHash);
    }
  }

  /**
   * The four-component form, kept so that existing callers compile unchanged and gain a hash
   * derived from their text.
   */
  public Document(String id, float[] vector, String text, Map<String, MetadataValue> metadata) {
    this(id, vector, text, metadata, null);
  }

  /**
   * A document whose source is deliberately not stored, carrying its hash instead.
   *
   * <p>For content that lives in a system of record, or is not text at all. Everything except
   * re-embedding from the collection alone still works: staleness, audit and lineage all rest on
   * the hash rather than on the bytes.
   */
  public static Document withHash(
      String id, float[] vector, String contentHash, Map<String, MetadataValue> metadata) {
    return new Document(
        id, vector, null, metadata, Objects.requireNonNull(contentHash, "contentHash"));
  }

  /** Whether this document's stored content still matches what it was embedded from. */
  public boolean matchesContent(String embeddedInput) {
    return contentHash != null && contentHash.equals(ContentHash.of(embeddedInput));
  }

  /** Whether this document carries provenance at all. */
  public boolean hasProvenance() {
    return contentHash != null;
  }

  /** Factory for an id + vector only. */
  public static Document of(String id, float[] vector) {
    return new Document(id, vector, null, Map.of());
  }

  /** Factory for an id + vector + text. */
  public static Document of(String id, float[] vector, String text) {
    return new Document(id, vector, text, Map.of());
  }
}
