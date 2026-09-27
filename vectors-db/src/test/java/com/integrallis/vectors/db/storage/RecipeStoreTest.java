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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The eight cases from the persistence proposal. Each one is a way the feature could silently lie
 * rather than fail, which is why they are written out individually instead of as one happy path.
 */
class RecipeStoreTest {

  private static EmbeddingRecipe recipe() {
    return new EmbeddingRecipe(
        "nomic-embed-text",
        "1.5",
        Optional.of("c".repeat(64)),
        768,
        SimilarityFunction.COSINE,
        true,
        EmbeddingRecipe.Pooling.MEAN,
        Optional.of("search_document: "),
        Optional.of("search_query: "),
        8192,
        EmbeddingRecipe.Truncation.REJECT,
        Map.of("tokenizer", "bert-base", "note", "quoted \"value\" with \\ and \n newline"));
  }

  @Test
  @DisplayName("1. every field round-trips and the hash survives a write/read cycle")
  void roundTrip(@TempDir Path root) throws IOException {
    EmbeddingRecipe original = recipe();
    RecipeStore.write(root, original);
    RecipeStore.Stored read =
        RecipeStore.read(root, Optional.of(original.recipeHash())).orElseThrow();
    assertEquals(original, read.recipe(), "a field was lost or mangled in serialisation");
    assertEquals(original.recipeHash(), read.recipe().recipeHash());
    assertTrue(read.hashVerifiable());
  }

  @Test
  @DisplayName("1b. an empty extra map and absent prefixes round-trip too")
  void roundTripMinimal(@TempDir Path root) throws IOException {
    EmbeddingRecipe minimal =
        EmbeddingRecipe.declared(
            "text-embedding-3-small", "2024-01", 1536, SimilarityFunction.COSINE);
    RecipeStore.write(root, minimal);
    assertEquals(
        minimal, RecipeStore.read(root, Optional.of(minimal.recipeHash())).orElseThrow().recipe());
  }

  @Test
  @DisplayName("2. a one-byte edit to the sidecar is detected, naming both hashes")
  void corruptedSidecarIsDetected(@TempDir Path root) throws IOException {
    EmbeddingRecipe original = recipe();
    RecipeStore.write(root, original);
    Path file = root.resolve(FileFormat.RECIPE_FILE);
    // Change the model version. The hash line in the file still says the old value, which is
    // exactly
    // the tampering this must catch -- and it must catch it by recomputing, not by trusting that
    // line.
    Files.writeString(
        file, Files.readString(file).replace("\"1.5\"", "\"1.6\""), StandardCharsets.UTF_8);

    IOException failure =
        assertThrows(
            IOException.class, () -> RecipeStore.read(root, Optional.of(original.recipeHash())));
    assertTrue(failure.getMessage().contains(original.recipeHash()), failure.getMessage());
    assertTrue(failure.getMessage().contains("altered or replaced"), failure.getMessage());
  }

  @Test
  @DisplayName("3. a declared recipe with no sidecar refuses, rather than degrading quietly")
  void missingSidecarWithAnchorRefuses(@TempDir Path root) {
    IOException failure =
        assertThrows(IOException.class, () -> RecipeStore.read(root, Optional.of("d".repeat(64))));
    assertTrue(failure.getMessage().contains("claims provenance"), failure.getMessage());
  }

  @Test
  @DisplayName("4. an orphan sidecar with no anchor is ignored, not honoured")
  void orphanSidecarIsIgnored(@TempDir Path root) throws IOException {
    RecipeStore.write(root, recipe());
    // No anchor in the manifest: this is the crashed-write case, or a file someone dropped in.
    // Honouring it would be how an accident or an attacker injects a recipe.
    assertTrue(RecipeStore.read(root, Optional.empty()).isEmpty());
  }

  @Test
  @DisplayName("5. a crash between the two writes leaves a collection that opens as UNKNOWN")
  void crashBetweenWritesIsRecoverable(@TempDir Path root) throws IOException {
    // Sidecar written, manifest not yet -- so no anchor exists.
    RecipeStore.write(root, recipe());
    assertTrue(
        RecipeStore.read(root, Optional.empty()).isEmpty(),
        "the recoverable case must be the one that happens: inert orphan, not a refusal to open");
  }

  @Test
  @DisplayName("6. a newer schema is readable for display but NOT attestable")
  void forwardCompatibilityWithholdsAttestation(@TempDir Path root) throws IOException {
    // The most valuable test here. A future build adds a field; this build must neither cry
    // tampering
    // (a false alarm) nor attest over fields it cannot see (a false assurance, which is worse).
    EmbeddingRecipe original = recipe();
    RecipeStore.write(root, original);
    Path file = root.resolve(FileFormat.RECIPE_FILE);
    Files.writeString(
        file,
        Files.readString(file)
            .replace("\"schemaVersion\": 1", "\"schemaVersion\": 2")
            .replace(
                "  \"modelId\":",
                "  \"futureField\": \"something this build cannot hash\",\n  \"modelId\":"),
        StandardCharsets.UTF_8);

    RecipeStore.Stored read =
        RecipeStore.read(root, Optional.of(original.recipeHash())).orElseThrow();
    assertEquals(2, read.schemaVersion());
    assertFalse(
        read.hashVerifiable(),
        "a newer schema cannot be attested by this build, because the canonical form it would hash "
            + "omits fields the file contains");
    assertEquals("nomic-embed-text", read.recipe().modelId(), "still readable for display");
  }

  @Test
  @DisplayName("8. a hand-edited dimension is caught by the hash, not silently accepted")
  void editedDimensionIsCaught(@TempDir Path root) throws IOException {
    EmbeddingRecipe original = recipe();
    RecipeStore.write(root, original);
    Path file = root.resolve(FileFormat.RECIPE_FILE);
    Files.writeString(
        file,
        Files.readString(file).replace("\"dimension\": 768", "\"dimension\": 384"),
        StandardCharsets.UTF_8);
    assertThrows(
        IOException.class, () -> RecipeStore.read(root, Optional.of(original.recipeHash())));
  }

  @Test
  @DisplayName("the stored hash line is a convenience and is never trusted on read")
  void storedHashLineIsNotTrusted(@TempDir Path root) throws IOException {
    EmbeddingRecipe original = recipe();
    RecipeStore.write(root, original);
    Path file = root.resolve(FileFormat.RECIPE_FILE);
    // Rewrite only the convenience hash line to garbage, leaving every field intact. Because read()
    // recomputes from the fields, this must still verify -- otherwise the file would be
    // self-attesting,
    // which is no attestation at all.
    Files.writeString(
        file,
        Files.readString(file).replace(original.recipeHash(), "0".repeat(64)),
        StandardCharsets.UTF_8);
    assertTrue(
        RecipeStore.read(root, Optional.of(original.recipeHash())).orElseThrow().hashVerifiable());
  }
}
