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
package com.integrallis.vectors.db.storage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@code recipe.json} must be byte-for-byte deterministic.
 *
 * <p>Not cosmetic. The file is an identity that gets diffed between environments, checked into
 * change control, and read by an auditor deciding whether two collections were built the same way.
 * A file whose bytes wobble with map iteration order or the host's line separator cannot serve any
 * of those.
 */
class RecipeJsonDeterminismTest {

  private static EmbeddingRecipe recipe(Map<String, String> extra) {
    return new EmbeddingRecipe(
        "nomic-embed-text",
        "1.5",
        Optional.of("f".repeat(64)),
        768,
        SimilarityFunction.COSINE,
        true,
        EmbeddingRecipe.Pooling.MEAN,
        Optional.of("search_document: "),
        Optional.of("search_query: "),
        8192,
        EmbeddingRecipe.Truncation.REJECT,
        extra);
  }

  @Test
  @DisplayName("serialising the same recipe twice produces identical bytes")
  void repeatedSerialisationIsIdentical() throws IOException {
    EmbeddingRecipe r = recipe(Map.of("tokenizer", "bert-base", "revision", "abc123"));
    assertEquals(RecipeStore.toJson(r), RecipeStore.toJson(r));
  }

  @Test
  @DisplayName("extra-map insertion order does not change a single byte")
  void mapOrderDoesNotLeakIntoTheFile() throws IOException {
    Map<String, String> forward = new LinkedHashMap<>();
    forward.put("alpha", "1");
    forward.put("beta", "2");
    forward.put("gamma", "3");
    Map<String, String> reverse = new LinkedHashMap<>();
    reverse.put("gamma", "3");
    reverse.put("beta", "2");
    reverse.put("alpha", "1");

    assertEquals(
        RecipeStore.toJson(recipe(forward)),
        RecipeStore.toJson(recipe(reverse)),
        "the extra map must be ordered by key on the way out, or two equal recipes differ on disk");
  }

  @Test
  @DisplayName("a TreeMap and a LinkedHashMap of the same entries agree")
  void mapImplementationDoesNotLeak() throws IOException {
    Map<String, String> linked = new LinkedHashMap<>(Map.of("b", "2", "a", "1"));
    Map<String, String> sorted = new TreeMap<>(Map.of("a", "1", "b", "2"));
    assertEquals(RecipeStore.toJson(recipe(linked)), RecipeStore.toJson(recipe(sorted)));
  }

  @Test
  @DisplayName("line endings are pinned to \\n, not the platform separator")
  void lineEndingsArePinned() throws IOException {
    // Jackson's DefaultPrettyPrinter uses SYSTEM_LINEFEED by default, so without an explicit
    // indenter the same recipe would produce CRLF on Windows and LF elsewhere -- two different
    // files
    // for one recipe, which defeats diffing it across environments.
    String json = RecipeStore.toJson(recipe(Map.of("k", "v")));
    assertTrue(json.contains("\n"), "expected newlines in the pretty-printed form");
    assertFalse(json.contains("\r"), "carriage returns make the file platform-dependent");
  }

  @Test
  @DisplayName("every field is written, so an absent field never means an unset one")
  void nullsAreWrittenNotOmitted() throws IOException {
    // A minimal recipe has no digest and no prefixes. Those keys must still appear, as null: if
    // they
    // were omitted, a reader could not distinguish "this writer had no digest" from "this writer
    // did
    // not know about digests".
    String json =
        RecipeStore.toJson(EmbeddingRecipe.declared("m", "1", 768, SimilarityFunction.COSINE));
    assertTrue(json.contains("\"modelDigest\" : null"), json);
    assertTrue(json.contains("\"documentPrefix\" : null"), json);
    assertTrue(json.contains("\"queryPrefix\" : null"), json);
  }

  @Test
  @DisplayName("field order is stable across serialisations")
  void fieldOrderIsStable() throws IOException {
    String first = RecipeStore.toJson(recipe(Map.of("z", "1")));
    String second = RecipeStore.toJson(recipe(Map.of("z", "1")));
    assertEquals(
        first.indexOf("modelId") < first.indexOf("dimension"),
        second.indexOf("modelId") < second.indexOf("dimension"));
    assertEquals(first, second);
  }

  @Test
  @DisplayName("a round trip through JSON preserves the recipe and therefore its hash")
  void roundTripPreservesTheHash() throws IOException {
    EmbeddingRecipe original = recipe(Map.of("tokenizer", "bert-base"));
    EmbeddingRecipe parsed = RecipeStore.parse(RecipeStore.toJson(original));
    assertEquals(original, parsed);
    assertEquals(original.recipeHash(), parsed.recipeHash());
  }

  @Test
  @DisplayName("an unknown future field does not break parsing")
  void unknownFieldsAreTolerated() throws IOException {
    EmbeddingRecipe original = recipe(Map.of());
    String json =
        RecipeStore.toJson(original)
            .replace("  \"modelId\"", "  \"fieldFromTheFuture\" : 42,\n  \"modelId\"");
    // Failing here would turn a forward-compatible file into an unopenable collection; attestation
    // is
    // withheld in read() instead.
    assertEquals(original.modelId(), RecipeStore.parse(json).modelId());
  }
}
