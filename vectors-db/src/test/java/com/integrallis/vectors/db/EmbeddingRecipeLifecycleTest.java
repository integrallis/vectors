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
package com.integrallis.vectors.db;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.core.ContentHash;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.FileFormat;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The recipe's lifecycle through a real collection: attach, persist, reopen, refuse. */
class EmbeddingRecipeLifecycleTest {

  private static EmbeddingRecipe recipe(String version) {
    return new EmbeddingRecipe(
        "nomic-embed-text",
        version,
        Optional.of("e".repeat(64)),
        4,
        SimilarityFunction.COSINE,
        true,
        EmbeddingRecipe.Pooling.MEAN,
        Optional.of("search_document: "),
        Optional.of("search_query: "),
        8192,
        EmbeddingRecipe.Truncation.REJECT,
        Map.of());
  }

  private static VectorCollectionBuilder base(Path root) {
    return VectorCollection.builder()
        .dimension(4)
        .metric(SimilarityFunction.COSINE)
        .storagePath(root);
  }

  @Test
  @DisplayName("a recipe survives a commit and a reopen, and the sidecar lands on disk")
  void recipeRoundTripsThroughACollection(@TempDir Path root) throws Exception {
    EmbeddingRecipe original = recipe("1.5");
    try (VectorCollection collection = base(root).embeddingRecipe(original).build()) {
      collection.add(new Document("a", new float[] {1, 0, 0, 0}, "alpha", Map.of()));
      collection.commit();
    }
    assertTrue(
        Files.exists(root.resolve(FileFormat.RECIPE_FILE)),
        "the sidecar must be written at the collection root");

    // Reopened WITHOUT repeating the recipe: it must be recovered, not silently dropped.
    try (VectorCollection reopened = base(root).build()) {
      assertEquals(
          Optional.of(original.recipeHash()),
          reopened.config().recipe().map(EmbeddingRecipe::recipeHash),
          "reopening without restating the recipe must keep it");
    }
  }

  @Test
  @DisplayName("a recipe makes provenance mandatory: a bare vector is refused")
  void recipeMakesProvenanceMandatory(@TempDir Path root) throws Exception {
    try (VectorCollection collection = base(root).embeddingRecipe(recipe("1.5")).build()) {
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () -> collection.add(Document.of("bare", new float[] {1, 0, 0, 0})));
      assertTrue(failure.getMessage().contains("stale"), failure.getMessage());

      // Either form of provenance is accepted.
      collection.add(new Document("withText", new float[] {0, 1, 0, 0}, "text", Map.of()));
      collection.add(
          Document.withHash(
              "withHash", new float[] {0, 0, 1, 0}, ContentHash.of("held elsewhere"), Map.of()));
    }
  }

  @Test
  @DisplayName("a collection with no recipe still accepts bare vectors")
  void noRecipeMeansNoObligation(@TempDir Path root) throws Exception {
    try (VectorCollection collection = base(root).build()) {
      collection.add(Document.of("bare", new float[] {1, 0, 0, 0}));
      collection.commit();
      assertTrue(collection.config().recipe().isEmpty());
    }
  }

  @Test
  @DisplayName("reopening with a DIFFERENT recipe is refused, naming both")
  void changingTheRecipeIsRefused(@TempDir Path root) throws Exception {
    try (VectorCollection collection = base(root).embeddingRecipe(recipe("1.5")).build()) {
      collection.add(new Document("a", new float[] {1, 0, 0, 0}, "alpha", Map.of()));
      collection.commit();
    }
    IllegalStateException failure =
        assertThrows(
            IllegalStateException.class, () -> base(root).embeddingRecipe(recipe("1.6")).build());
    assertTrue(failure.getMessage().contains("migration"), failure.getMessage());
    assertTrue(failure.getMessage().contains("1.5"), failure.getMessage());
    assertTrue(failure.getMessage().contains("1.6"), failure.getMessage());
  }

  @Test
  @DisplayName("attaching a recipe to a collection that already holds vectors is refused")
  void cannotAttachARecipeRetroactively(@TempDir Path root) throws Exception {
    try (VectorCollection collection = base(root).build()) {
      collection.add(Document.of("a", new float[] {1, 0, 0, 0}));
      collection.commit();
    }
    // Those vectors were not produced by this recipe, so recording it would attest to something
    // untrue about every one of them.
    IllegalStateException failure =
        assertThrows(
            IllegalStateException.class, () -> base(root).embeddingRecipe(recipe("1.5")).build());
    assertTrue(failure.getMessage().contains("already holds vectors"), failure.getMessage());
  }

  @Test
  @DisplayName("a recipe whose dimension contradicts the collection is refused at construction")
  void mismatchedRecipeDimensionIsRefused(@TempDir Path root) {
    EmbeddingRecipe wrongDimension =
        new EmbeddingRecipe(
            "m",
            "1",
            Optional.empty(),
            768,
            SimilarityFunction.COSINE,
            true,
            EmbeddingRecipe.Pooling.MEAN,
            Optional.empty(),
            Optional.empty(),
            512,
            EmbeddingRecipe.Truncation.REJECT,
            Map.of());
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> base(root).embeddingRecipe(wrongDimension).build());
    assertTrue(failure.getMessage().contains("untrue"), failure.getMessage());
  }

  @Test
  @DisplayName("the manifest anchors the recipe hash, so the sidecar cannot be swapped unnoticed")
  void manifestAnchorsTheRecipe(@TempDir Path root) throws Exception {
    EmbeddingRecipe original = recipe("1.5");
    try (VectorCollection collection = base(root).embeddingRecipe(original).build()) {
      collection.add(new Document("a", new float[] {1, 0, 0, 0}, "alpha", Map.of()));
      collection.commit();
    }
    Path sidecar = root.resolve(FileFormat.RECIPE_FILE);
    Files.writeString(sidecar, Files.readString(sidecar).replace("\"1.5\"", "\"9.9\""));

    // The manifest still anchors the original hash, so the edit must be detected on reopen.
    Exception failure = assertThrows(Exception.class, () -> base(root).build());
    assertTrue(
        failure.getMessage() != null
            && (failure.getMessage().contains("altered")
                || failure.getMessage().contains("recipe")),
        "expected the swap to be detected, got: " + failure.getMessage());
  }
}
