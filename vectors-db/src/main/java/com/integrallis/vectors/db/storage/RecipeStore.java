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
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
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
    int schemaVersion = intField(json, "schemaVersion");
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

  // --- serialisation. Hand-rolled so vectors-db gains no JSON dependency for one small file. ---

  static String toJson(EmbeddingRecipe recipe) {
    StringBuilder out = new StringBuilder(512);
    out.append("{\n  \"schemaVersion\": ").append(SCHEMA_VERSION).append(",\n");
    // The hash is written for an auditor's convenience. It is NOT trusted on read: read()
    // recomputes
    // it from the fields, so editing this line changes nothing and editing a field is detected.
    out.append("  \"recipeHash\": \"").append(recipe.recipeHash()).append("\",\n");
    out.append("  \"modelId\": ").append(quote(recipe.modelId())).append(",\n");
    out.append("  \"modelVersion\": ").append(quote(recipe.modelVersion())).append(",\n");
    out.append("  \"modelDigest\": ")
        .append(recipe.modelDigest().map(RecipeStore::quote).orElse("null"))
        .append(",\n");
    out.append("  \"dimension\": ").append(recipe.dimension()).append(",\n");
    out.append("  \"metric\": ").append(quote(recipe.metric().name())).append(",\n");
    out.append("  \"normalized\": ").append(recipe.normalized()).append(",\n");
    out.append("  \"pooling\": ").append(quote(recipe.pooling().name())).append(",\n");
    out.append("  \"documentPrefix\": ")
        .append(recipe.documentPrefix().map(RecipeStore::quote).orElse("null"))
        .append(",\n");
    out.append("  \"queryPrefix\": ")
        .append(recipe.queryPrefix().map(RecipeStore::quote).orElse("null"))
        .append(",\n");
    out.append("  \"maxInputTokens\": ").append(recipe.maxInputTokens()).append(",\n");
    out.append("  \"truncation\": ").append(quote(recipe.truncation().name())).append(",\n");
    out.append("  \"extra\": {");
    Map<String, String> sorted = new TreeMap<>(recipe.extra());
    boolean first = true;
    for (Map.Entry<String, String> entry : sorted.entrySet()) {
      if (!first) {
        out.append(',');
      }
      first = false;
      out.append("\n    ")
          .append(quote(entry.getKey()))
          .append(": ")
          .append(quote(entry.getValue()));
    }
    out.append(sorted.isEmpty() ? "}\n" : "\n  }\n").append("}\n");
    return out.toString();
  }

  static EmbeddingRecipe parse(String json) throws IOException {
    try {
      Map<String, String> extra = new LinkedHashMap<>();
      int extraStart = json.indexOf("\"extra\": {");
      if (extraStart >= 0) {
        String block = json.substring(extraStart + 10, json.indexOf('}', extraStart));
        for (String pair : block.split(",")) {
          int colon = pair.indexOf("\":");
          if (colon > 0) {
            extra.put(
                unquote(pair.substring(0, colon + 1).trim()),
                unquote(pair.substring(colon + 2).trim()));
          }
        }
      }
      return new EmbeddingRecipe(
          stringField(json, "modelId"),
          stringField(json, "modelVersion"),
          optionalField(json, "modelDigest"),
          intField(json, "dimension"),
          SimilarityFunction.valueOf(stringField(json, "metric")),
          Boolean.parseBoolean(rawField(json, "normalized")),
          EmbeddingRecipe.Pooling.valueOf(stringField(json, "pooling")),
          optionalField(json, "documentPrefix"),
          optionalField(json, "queryPrefix"),
          intField(json, "maxInputTokens"),
          EmbeddingRecipe.Truncation.valueOf(stringField(json, "truncation")),
          extra);
    } catch (RuntimeException malformed) {
      throw new IOException(
          "Malformed " + FileFormat.RECIPE_FILE + ": " + malformed.getMessage(), malformed);
    }
  }

  private static String quote(String value) {
    StringBuilder out = new StringBuilder(value.length() + 2).append('"');
    for (int index = 0; index < value.length(); index++) {
      char c = value.charAt(index);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.append('"').toString();
  }

  private static String unquote(String quoted) {
    String trimmed = quoted.trim();
    if (trimmed.startsWith("\"")) {
      trimmed = trimmed.substring(1);
    }
    if (trimmed.endsWith("\"")) {
      trimmed = trimmed.substring(0, trimmed.length() - 1);
    }
    return trimmed
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\");
  }

  private static String rawField(String json, String name) {
    int at = json.indexOf('"' + name + "\": ");
    if (at < 0) {
      throw new IllegalArgumentException("missing field " + name);
    }
    int start = at + name.length() + 4;
    int end = start;
    while (end < json.length() && json.charAt(end) != ',' && json.charAt(end) != '\n') {
      end++;
    }
    return json.substring(start, end).trim();
  }

  private static String stringField(String json, String name) {
    return unquote(rawField(json, name));
  }

  private static Optional<String> optionalField(String json, String name) {
    String raw = rawField(json, name);
    return "null".equals(raw) ? Optional.empty() : Optional.of(unquote(raw));
  }

  private static int intField(String json, String name) {
    return Integer.parseInt(rawField(json, name));
  }
}
