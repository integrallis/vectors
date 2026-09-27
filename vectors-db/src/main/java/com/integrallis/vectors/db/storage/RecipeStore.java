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

import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.RecipeCodec;
import com.integrallis.vectors.core.RecipeCodecs;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.Optional;

/**
 * Reads and writes {@code recipe.json}, the body whose hash the manifest anchors.
 *
 * <h2>Write ordering, and why it is this way round</h2>
 *
 * <p>The sidecar is written and fsynced <b>before</b> the manifest that anchors it. A crash between
 * the two therefore leaves an <b>orphan sidecar</b>, which every reader ignores because no manifest
 * references it — harmless, and cleaned up by the next successful write. The reverse order would
 * leave a manifest anchoring a recipe that does not exist, which is a collection that refuses to
 * open. Order the failure so that the recoverable case is the one that happens.
 *
 * <h2>Schema evolution</h2>
 *
 * <p>{@link #SCHEMA_VERSION} is written into the file. A reader encountering a <em>higher</em>
 * version must not hash the fields it happens to understand — that would report tampering on a file
 * that is merely newer — and must not ignore the unknown fields and attest anyway, which is worse
 * because it asserts something unverified. It reports the model for display and withholds
 * attestation. See {@link #read}.
 */
public final class RecipeStore {

  /**
   * The codec in use, resolved once.
   *
   * <p>Pluggable so an application can use the JSON stack it already standardises on; the default
   * carries no dependency. Safe to swap because {@link EmbeddingRecipe#recipeHash()} is computed
   * from a canonical field rendering rather than the serialised text, so a collection written by
   * one codec verifies under another.
   */
  private static final RecipeCodec CODEC = RecipeCodecs.discover();

  /** Schema version this build writes. */
  public static final int SCHEMA_VERSION = CODEC.schemaVersion();

  private RecipeStore() {}

  /** A recipe as read from disk, with whether this build can vouch for its hash. */
  public record Stored(EmbeddingRecipe recipe, int schemaVersion, boolean hashVerifiable) {}

  /**
   * Writes the recipe, fsyncing the file and the directory entry before returning.
   *
   * <p>Call this <b>before</b> writing the manifest that anchors it.
   */
  public static void write(Path collectionRoot, EmbeddingRecipe recipe) throws IOException {
    Objects.requireNonNull(collectionRoot, "collectionRoot");
    Objects.requireNonNull(recipe, "recipe");
    Path tmp = collectionRoot.resolve(FileFormat.RECIPE_TMP_FILE);
    Path target = collectionRoot.resolve(FileFormat.RECIPE_FILE);
    byte[] body = CODEC.encode(recipe).getBytes(StandardCharsets.UTF_8);

    Files.write(
        tmp,
        body,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    try (FileChannel channel = FileChannel.open(tmp, StandardOpenOption.WRITE)) {
      channel.force(true);
    }
    Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    // The rename is only durable once the directory entry is too.
    try (FileChannel dir = FileChannel.open(collectionRoot, StandardOpenOption.READ)) {
      dir.force(true);
    } catch (IOException ignored) {
      // Some filesystems refuse to open a directory for fsync. The atomic rename still holds; this
      // only weakens durability across a power loss, so it is not worth failing a write over.
    }
  }

  /**
   * Reads the recipe, if present.
   *
   * @param anchoredHash the hash recorded in the manifest, or empty if the manifest declares none
   * @throws IOException if the manifest and the sidecar disagree about presence or identity
   */
  public static Optional<Stored> read(Path collectionRoot, Optional<String> anchoredHash)
      throws IOException {
    Path target = collectionRoot.resolve(FileFormat.RECIPE_FILE);
    boolean present = Files.exists(target);

    if (anchoredHash.isEmpty()) {
      if (present) {
        // An orphan from a crashed write, or a file someone dropped in. Ignoring it is deliberate:
        // honouring an unreferenced recipe is exactly how an accident or an attacker would inject
        // one.
        return Optional.empty();
      }
      return Optional.empty();
    }
    if (!present) {
      throw new IOException(
          "Collection manifest declares an embedding recipe (hash "
              + anchoredHash.get()
              + ") but "
              + FileFormat.RECIPE_FILE
              + " is missing. The collection claims provenance it cannot produce; restore the file or"
              + " rebuild the collection.");
    }

    String json = Files.readString(target, StandardCharsets.UTF_8);
    int schemaVersion = CODEC.schemaVersionOf(json);
    if (schemaVersion > SCHEMA_VERSION) {
      // Readable for display, but not attestable: the canonical form this build would hash omits
      // fields the file contains, so a computed hash would differ for a reason that is not
      // tampering.
      return Optional.of(new Stored(CODEC.decode(json), schemaVersion, false));
    }
    EmbeddingRecipe recipe = CODEC.decode(json);
    String computed = recipe.recipeHash();
    if (!computed.equals(anchoredHash.get())) {
      throw new IOException(
          "Embedding recipe does not match the hash anchored in the manifest.\n  manifest: "
              + anchoredHash.get()
              + "\n  computed: "
              + computed
              + "\nThe recipe file has been altered or replaced since the collection was written.");
    }
    return Optional.of(new Stored(recipe, schemaVersion, true));
  }

  /** Removes the sidecar, used when a write is abandoned. */
  public static void deleteIfPresent(Path collectionRoot) throws IOException {
    Files.deleteIfExists(collectionRoot.resolve(FileFormat.RECIPE_FILE));
    Files.deleteIfExists(collectionRoot.resolve(FileFormat.RECIPE_TMP_FILE));
  }

  // --- serialisation lives behind RecipeCodec; see BuiltinRecipeCodec and vectors-db-jackson ---

  /** The codec this build resolved, for diagnostics and tests. */
  public static RecipeCodec codec() {
    return CODEC;
  }
}
