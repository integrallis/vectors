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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.EmbeddingRecipe.Attestation;
import com.integrallis.vectors.core.EmbeddingRecipe.Pooling;
import com.integrallis.vectors.core.EmbeddingRecipe.Truncation;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Provenance is only useful if it cannot be absent, cannot drift, and answers staleness. */
class EmbeddingProvenanceTest {

  private static EmbeddingRecipe recipe(Map<String, String> extra) {
    return new EmbeddingRecipe(
        "nomic-embed-text",
        "1.5",
        Optional.of("a".repeat(64)),
        768,
        SimilarityFunction.COSINE,
        true,
        Pooling.MEAN,
        Optional.of("search_document: "),
        Optional.of("search_query: "),
        8192,
        Truncation.REJECT,
        extra);
  }

  @Test
  @DisplayName("a document always has a hash, derived from its text when one is given")
  void hashIsDerivedFromText() {
    Document document = new Document("d1", new float[] {1, 2}, "hello", Map.of());
    assertEquals(ContentHash.of("hello"), document.contentHash());
    assertEquals(ContentHash.LENGTH, document.contentHash().length());
  }

  @Test
  @DisplayName("a bare vector is allowed and simply carries no provenance")
  void bareVectorIsAllowed() {
    // Deliberate, and learned from the test suite: Document.of(id, vector) is pervasive --
    // compaction fixtures, generation shipping, and studio-web re-wrapping vectors that already
    // exist in a collection. Those vectors have no content to hash, so refusing them would break
    // plain vector storage to enforce a rule that does not apply to it.
    Document document = Document.of("d1", new float[] {1, 2});
    assertFalse(document.hasProvenance());
    assertNull(document.contentHash());
  }

  @Test
  @DisplayName("a recipe refuses a document with no provenance, and says why")
  void recipeEnforcesProvenanceWhereContentIsClaimed() {
    // The obligation belongs here, not in Document: declaring HOW vectors are made is what creates
    // the duty to record WHAT each was made from.
    EmbeddingRecipe declared = recipe(Map.of());
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> declared.requireProvenance(Document.of("d1", new float[] {1, 2})));
    assertTrue(failure.getMessage().contains("stale"), failure.getMessage());
    assertTrue(failure.getMessage().contains("withHash"), failure.getMessage());
    assertTrue(failure.getMessage().contains("d1"), "the message must name the document");
  }

  @Test
  @DisplayName("a recipe accepts a document that carries provenance, by text or by hash")
  void recipeAcceptsProvenancedDocuments() {
    EmbeddingRecipe declared = recipe(Map.of());
    declared.requireProvenance(new Document("d1", new float[] {1}, "hello", Map.of()));
    declared.requireProvenance(
        Document.withHash("d2", new float[] {1}, ContentHash.of("elsewhere"), Map.of()));
  }

  @Test
  @DisplayName("source may be omitted deliberately, in which case the hash carries the provenance")
  void hashWithoutSourceIsAllowed() {
    String hash = ContentHash.of("content that lives in S3");
    Document document = Document.withHash("d1", new float[] {1, 2}, hash, Map.of());
    assertEquals(hash, document.contentHash());
    assertEquals(null, document.text());
  }

  @Test
  @DisplayName("a malformed supplied hash is rejected, not stored")
  void malformedHashIsRejected() {
    assertThrows(
        IllegalArgumentException.class,
        () -> Document.withHash("d1", new float[] {1}, "deadbeef", Map.of()));
    assertThrows(
        IllegalArgumentException.class,
        () -> Document.withHash("d1", new float[] {1}, "A".repeat(64), Map.of()),
        "uppercase hex must be refused, or two encodings of one hash would both exist");
  }

  @Test
  @DisplayName("staleness is decidable: changed content stops matching")
  void stalenessIsDecidable() {
    Document document = new Document("d1", new float[] {1}, "version one", Map.of());
    assertTrue(document.matchesContent("version one"));
    assertFalse(document.matchesContent("version two"));
  }

  @Test
  @DisplayName("the recipe's prefixes are asymmetric, which is the point of holding them centrally")
  void documentAndQueryPrefixesDiffer() {
    EmbeddingRecipe r = recipe(Map.of());
    assertEquals("search_document: cats", r.documentInput("cats"));
    assertEquals("search_query: cats", r.queryInput("cats"));
    assertNotEquals(r.documentInput("cats"), r.queryInput("cats"));
    // And therefore the hash of what was embedded is not the hash of the bare text.
    assertNotEquals(ContentHash.of("cats"), ContentHash.of(r.documentInput("cats")));
  }

  @Test
  @DisplayName("recipeHash is stable across map ordering, so it can be compared across versions")
  void recipeHashIsOrderStable() {
    Map<String, String> one = new LinkedHashMap<>();
    one.put("alpha", "1");
    one.put("beta", "2");
    Map<String, String> other = new LinkedHashMap<>();
    other.put("beta", "2");
    other.put("alpha", "1");
    assertEquals(recipe(one).recipeHash(), recipe(other).recipeHash());
  }

  @Test
  @DisplayName("every component that changes the vector changes the recipe hash")
  void recipeHashCoversEverythingThatMatters() {
    EmbeddingRecipe base = recipe(Map.of());
    String baseline = base.recipeHash();

    // A prefix change leaves the source untouched and every vector different -- the case a
    // content-only hash cannot see.
    assertNotEquals(
        baseline,
        new EmbeddingRecipe(
                base.modelId(),
                base.modelVersion(),
                base.modelDigest(),
                base.dimension(),
                base.metric(),
                base.normalized(),
                base.pooling(),
                Optional.of("passage: "),
                base.queryPrefix(),
                base.maxInputTokens(),
                base.truncation(),
                base.extra())
            .recipeHash(),
        "a changed document prefix must change the recipe hash");

    assertNotEquals(
        baseline,
        new EmbeddingRecipe(
                base.modelId(),
                base.modelVersion(),
                base.modelDigest(),
                base.dimension(),
                base.metric(),
                base.normalized(),
                Pooling.CLS,
                base.documentPrefix(),
                base.queryPrefix(),
                base.maxInputTokens(),
                base.truncation(),
                base.extra())
            .recipeHash(),
        "a changed pooling must change the recipe hash");

    assertNotEquals(
        baseline,
        new EmbeddingRecipe(
                base.modelId(),
                base.modelVersion(),
                Optional.of("b".repeat(64)),
                base.dimension(),
                base.metric(),
                base.normalized(),
                base.pooling(),
                base.documentPrefix(),
                base.queryPrefix(),
                base.maxInputTokens(),
                base.truncation(),
                base.extra())
            .recipeHash(),
        "a different model digest is a different model, whatever its name says");
  }

  @Test
  @DisplayName("length prefixing stops adjacent fields colliding")
  void fieldsCannotRunTogether() {
    // Without length prefixes, ("ab","c") and ("a","bc") would canonicalise identically.
    EmbeddingRecipe left = EmbeddingRecipe.declared("ab", "c", 768, SimilarityFunction.COSINE);
    EmbeddingRecipe right = EmbeddingRecipe.declared("a", "bc", 768, SimilarityFunction.COSINE);
    assertNotEquals(left.recipeHash(), right.recipeHash());
  }

  @Test
  @DisplayName("attestation distinguishes a verifiable model from one we were merely told about")
  void attestationIsNotOptimistic() {
    assertEquals(Attestation.ATTESTED, recipe(Map.of()).attestation());
    assertEquals(
        Attestation.DECLARED,
        EmbeddingRecipe.declared(
                "text-embedding-3-small", "2024-01", 1536, SimilarityFunction.COSINE)
            .attestation());
  }
}
