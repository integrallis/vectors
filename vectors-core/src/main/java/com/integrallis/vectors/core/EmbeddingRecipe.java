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
import java.util.Optional;
import java.util.TreeMap;

/**
 * Everything that decides what vector a piece of content becomes.
 *
 * <p>A collection's only tie to the model that filled it used to be the dimension, and dimension is
 * not identity: two unrelated 768-dimension models produce mutually meaningless vectors, and a
 * store that knows only the dimension mixes them without complaint. A query embedded with one and
 * searched against the other returns confidently ranked nonsense, with no error and no way to
 * notice afterwards.
 *
 * <p>Every component here changes the resulting vector, so omitting any of them would make the
 * recipe an incomplete identity. The one most often got wrong is {@link #documentPrefix} versus
 * {@link #queryPrefix}: instruction-tuned embedders are <b>asymmetric</b> — the same text embedded
 * as a document and as a query must carry different instructions and yields different vectors.
 * Putting that in the recipe rather than at each call site is what stops one caller getting it
 * wrong.
 *
 * <h2>Why this belongs to the collection, and why it is immutable</h2>
 *
 * <p>Changing a recipe invalidates every vector already stored, so it cannot be an update — it is a
 * migration. Weaviate learned this and makes its vectorizer config immutable for exactly this
 * reason.
 *
 * <p>That immutability is what lets a {@link Document} carry a <b>single</b> hash. Because every
 * vector in a collection came from the same recipe, a document is stale iff its own content
 * changed; there is no need to record a per-document recipe to work out which generation it belongs
 * to. Allow recipes to mutate in place and that stops being true, and every document has to carry
 * its own recipe identity.
 *
 * <h2>Attested versus declared</h2>
 *
 * <p>{@link #modelDigest} is what separates provenance you can check from provenance you were told.
 * A module name or a service endpoint records what you *said* you used: swap the model behind it
 * and the recipe is unchanged while every later vector differs. A digest of the actual weights — a
 * ModelJars coordinate and its SHA-256 — is verifiable. Where a hosted provider offers no digest
 * the recipe is still useful, but it is {@link Attestation#DECLARED} and must not be presented as
 * more.
 */
public record EmbeddingRecipe(
    String modelId,
    String modelVersion,
    Optional<String> modelDigest,
    int dimension,
    SimilarityFunction metric,
    boolean normalized,
    Pooling pooling,
    Optional<String> documentPrefix,
    Optional<String> queryPrefix,
    int maxInputTokens,
    Truncation truncation,
    Map<String, String> extra) {

  /** How token vectors are collapsed into one vector. Changes the result entirely. */
  public enum Pooling {
    MEAN,
    CLS,
    LAST_TOKEN,
    /** The model emits one vector directly; nothing to pool. */
    NONE
  }

  /** What happens to input longer than {@link #maxInputTokens}. */
  public enum Truncation {
    /** Silently keep the head. Records that the tail was never seen. */
    HEAD,
    /** Silently keep the tail. */
    TAIL,
    /** Refuse. The safest choice for a collection that must be reproducible. */
    REJECT
  }

  /** Whether the model identity can be verified or only believed. */
  public enum Attestation {
    /** A weights digest is recorded; a stored vector can be re-derived and compared. */
    ATTESTED,
    /** Only a name and version; a silent model swap behind that name is undetectable. */
    DECLARED
  }

  public EmbeddingRecipe {
    Objects.requireNonNull(modelId, "modelId");
    Objects.requireNonNull(modelVersion, "modelVersion");
    Objects.requireNonNull(modelDigest, "modelDigest");
    Objects.requireNonNull(metric, "metric");
    Objects.requireNonNull(pooling, "pooling");
    Objects.requireNonNull(documentPrefix, "documentPrefix");
    Objects.requireNonNull(queryPrefix, "queryPrefix");
    Objects.requireNonNull(truncation, "truncation");
    if (modelId.isBlank()) {
      throw new IllegalArgumentException("modelId must not be blank");
    }
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    if (maxInputTokens <= 0) {
      throw new IllegalArgumentException("maxInputTokens must be positive: " + maxInputTokens);
    }
    // Sorted and copied so that recipeHash() is stable regardless of insertion order.
    extra = extra == null ? Map.of() : Map.copyOf(new TreeMap<>(extra));
  }

  /** Whether this recipe's model identity is verifiable. */
  public Attestation attestation() {
    return modelDigest.isPresent() ? Attestation.ATTESTED : Attestation.DECLARED;
  }

  /**
   * A stable identity for this recipe, for deciding whether stored vectors are of this generation.
   *
   * <p>Canonical form is a sorted, explicitly delimited rendering rather than {@code toString} or
   * serialisation, because the hash is persisted and compared across versions: it must not move
   * when a field is reordered, a record's {@code toString} changes, or a map iterates differently.
   * Fields are length-prefixed so that no combination of values can collide by running together.
   */
  public String recipeHash() {
    StringBuilder canonical = new StringBuilder(256);
    append(canonical, "modelId", modelId);
    append(canonical, "modelVersion", modelVersion);
    append(canonical, "modelDigest", modelDigest.orElse(""));
    append(canonical, "dimension", Integer.toString(dimension));
    append(canonical, "metric", metric.name());
    append(canonical, "normalized", Boolean.toString(normalized));
    append(canonical, "pooling", pooling.name());
    append(canonical, "documentPrefix", documentPrefix.orElse(""));
    append(canonical, "queryPrefix", queryPrefix.orElse(""));
    append(canonical, "maxInputTokens", Integer.toString(maxInputTokens));
    append(canonical, "truncation", truncation.name());
    for (Map.Entry<String, String> entry : new TreeMap<>(extra).entrySet()) {
      append(canonical, "extra." + entry.getKey(), entry.getValue());
    }
    return ContentHash.of(canonical.toString());
  }

  private static void append(StringBuilder target, String key, String value) {
    target.append(key).append('=').append(value.length()).append(':').append(value).append('\n');
  }

  /**
   * The text that should actually be embedded for a document, with this recipe's prefix applied.
   *
   * <p>Use this rather than prefixing at the call site, and hash <em>its</em> result — that is what
   * makes {@link Document#contentHash()} answer "would re-embedding change the vector".
   */
  public String documentInput(String text) {
    Objects.requireNonNull(text, "text");
    return documentPrefix.map(prefix -> prefix + text).orElse(text);
  }

  /** The query counterpart of {@link #documentInput}. Deliberately a different prefix. */
  public String queryInput(String text) {
    Objects.requireNonNull(text, "text");
    return queryPrefix.map(prefix -> prefix + text).orElse(text);
  }

  /**
   * Checks that a document carries the provenance this recipe's collection requires.
   *
   * <p>A collection that declares how its vectors are made must know what each was made from,
   * otherwise it cannot say which are stale — which is the whole point of holding a recipe. A
   * collection with no recipe makes no such claim and imposes no such obligation, so using the
   * store as plain vector storage stays possible.
   *
   * @throws IllegalArgumentException if the document has no content hash
   */
  public void requireProvenance(Document document) {
    Objects.requireNonNull(document, "document");
    if (!document.hasProvenance()) {
      throw new IllegalArgumentException(
          "document '"
              + document.id()
              + "' has no contentHash, but this collection declares an embedding recipe ("
              + modelId
              + ' '
              + modelVersion
              + "). A collection that records how its vectors are produced must record what each was "
              + "produced from, or it cannot tell which vectors are stale. Supply text, or use "
              + "Document.withHash(...) when the source is held elsewhere.");
    }
  }

  /** A minimal declared recipe, for a hosted model that offers no weights digest. */
  public static EmbeddingRecipe declared(
      String modelId, String modelVersion, int dimension, SimilarityFunction metric) {
    return new EmbeddingRecipe(
        modelId,
        modelVersion,
        Optional.empty(),
        dimension,
        metric,
        true,
        Pooling.NONE,
        Optional.empty(),
        Optional.empty(),
        8192,
        Truncation.REJECT,
        Map.of());
  }
}
