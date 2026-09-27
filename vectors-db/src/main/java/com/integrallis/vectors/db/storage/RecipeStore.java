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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;

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

  /** Version of the {@code recipe.json} layout. Bump when a field is added. */
  public static final int SCHEMA_VERSION = 1;

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
    byte[] body = toJson(recipe).getBytes(StandardCharsets.UTF_8);

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
    int schemaVersion = schemaVersionOf(json);
    if (schemaVersion > SCHEMA_VERSION) {
      // Readable for display, but not attestable: the canonical form this build would hash omits
      // fields the file contains, so a computed hash would differ for a reason that is not
      // tampering.
      return Optional.of(new Stored(parse(json), schemaVersion, false));
    }
    EmbeddingRecipe recipe = parse(json);
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

  // --- serialisation: Jackson, with byte-for-byte determinism as an explicit requirement ---

  /**
   * Shared mapper, configured so the same recipe always produces the same bytes.
   *
   * <p>Determinism is not automatic and each setting below buys a specific part of it:
   *
   * <ul>
   *   <li>{@code ORDER_MAP_ENTRIES_BY_KEYS} — the {@code extra} map would otherwise serialise in
   *       whatever order its implementation iterates;
   *   <li>a {@link DefaultIndenter} pinned to {@code "\n"} — Jackson's default pretty printer uses
   *       {@code SYSTEM_LINEFEED}, so the same recipe would produce CRLF on Windows and LF
   *       elsewhere;
   *   <li>{@code NON_NULL} never applies — every field is written, including nulls, so a field's
   *       absence can never be confused with its being unset by an older writer;
   *   <li>field order comes from the record's declaration order, which Jackson preserves.
   * </ul>
   */
  private static final ObjectMapper MAPPER =
      JsonMapper.builder()
          .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
          .enable(SerializationFeature.INDENT_OUTPUT)
          .build();

  private static final DefaultPrettyPrinter PRINTER =
      new DefaultPrettyPrinter()
          .withObjectIndenter(new DefaultIndenter("  ", "\n"))
          .withArrayIndenter(new DefaultIndenter("  ", "\n"));

  /**
   * The on-disk shape. A DTO rather than serialising {@link EmbeddingRecipe} directly, so the file
   * format is decoupled from the domain record: a field can be added to one without silently
   * changing the other, and {@code schemaVersion} has somewhere to live.
   */
  record RecipeDocument(
      int schemaVersion,
      String recipeHash,
      String modelId,
      String modelVersion,
      String modelDigest,
      int dimension,
      String metric,
      boolean normalized,
      String pooling,
      String documentPrefix,
      String queryPrefix,
      int maxInputTokens,
      String truncation,
      Map<String, String> extra) {}

  static String toJson(EmbeddingRecipe recipe) throws IOException {
    RecipeDocument document =
        new RecipeDocument(
            SCHEMA_VERSION,
            // Written for an auditor's convenience. NOT trusted on read -- read() recomputes it
            // from
            // the fields, so editing this line changes nothing and editing a field is detected.
            recipe.recipeHash(),
            recipe.modelId(),
            recipe.modelVersion(),
            recipe.modelDigest().orElse(null),
            recipe.dimension(),
            recipe.metric().name(),
            recipe.normalized(),
            recipe.pooling().name(),
            recipe.documentPrefix().orElse(null),
            recipe.queryPrefix().orElse(null),
            recipe.maxInputTokens(),
            recipe.truncation().name(),
            new TreeMap<>(recipe.extra()));
    return MAPPER.writer(PRINTER).writeValueAsString(document) + "\n";
  }

  static EmbeddingRecipe parse(String json) throws IOException {
    RecipeDocument document = readDocument(json);
    return new EmbeddingRecipe(
        document.modelId(),
        document.modelVersion(),
        Optional.ofNullable(document.modelDigest()),
        document.dimension(),
        SimilarityFunction.valueOf(document.metric()),
        document.normalized(),
        EmbeddingRecipe.Pooling.valueOf(document.pooling()),
        Optional.ofNullable(document.documentPrefix()),
        Optional.ofNullable(document.queryPrefix()),
        document.maxInputTokens(),
        EmbeddingRecipe.Truncation.valueOf(document.truncation()),
        document.extra() == null ? Map.of() : document.extra());
  }

  private static RecipeDocument readDocument(String json) throws IOException {
    try {
      // Unknown properties are tolerated on purpose: a newer writer may have added fields, and
      // read()
      // withholds attestation in that case rather than failing. Failing here would turn a forward-
      // compatible file into an unopenable collection.
      return MAPPER
          .readerFor(RecipeDocument.class)
          .without(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
          .readValue(json);
    } catch (RuntimeException | JsonProcessingException malformed) {
      throw new IOException(
          "Malformed " + FileFormat.RECIPE_FILE + ": " + malformed.getMessage(), malformed);
    }
  }

  static int schemaVersionOf(String json) throws IOException {
    return readDocument(json).schemaVersion();
  }
}
