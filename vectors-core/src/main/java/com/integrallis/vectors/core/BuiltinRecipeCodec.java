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

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;

/**
 * The default {@link RecipeCodec}: JSON, with no external dependency.
 *
 * <p>Deliberately dependency-free. {@code recipe.json} is one small file, and forcing a JSON
 * library onto every consumer of the core library to write it would be a poor trade — particularly
 * for applications that standardise on a different stack and would then carry two. Put {@code
 * vectors-db-jackson} on the classpath to use Jackson instead, or supply your own codec; both are
 * discovered by {@link java.util.ServiceLoader}.
 *
 * <p>The output is deterministic by construction: fields are written in a fixed order, {@code
 * extra} is sorted by key, and the line separator is a literal {@code \n} rather than the
 * platform's.
 */
public final class BuiltinRecipeCodec implements RecipeCodec {

  /** Schema version of the layout this codec reads and writes. */
  public static final int SCHEMA_VERSION = 1;

  @Override
  public int schemaVersion() {
    return SCHEMA_VERSION;
  }

  @Override
  public String encode(EmbeddingRecipe recipe) {
    StringBuilder out = new StringBuilder(512).append("{\n");
    field(out, "schemaVersion", Integer.toString(SCHEMA_VERSION), false);
    // Written for an auditor's convenience and never trusted on read: the hash is recomputed from
    // the
    // fields, so editing this line changes nothing while editing a field is detected.
    field(out, "recipeHash", quote(recipe.recipeHash()), false);
    field(out, "modelId", quote(recipe.modelId()), false);
    field(out, "modelVersion", quote(recipe.modelVersion()), false);
    field(
        out,
        "modelDigest",
        recipe.modelDigest().map(BuiltinRecipeCodec::quote).orElse("null"),
        false);
    field(out, "dimension", Integer.toString(recipe.dimension()), false);
    field(out, "metric", quote(recipe.metric().name()), false);
    field(out, "normalized", Boolean.toString(recipe.normalized()), false);
    field(out, "pooling", quote(recipe.pooling().name()), false);
    field(
        out,
        "documentPrefix",
        recipe.documentPrefix().map(BuiltinRecipeCodec::quote).orElse("null"),
        false);
    field(
        out,
        "queryPrefix",
        recipe.queryPrefix().map(BuiltinRecipeCodec::quote).orElse("null"),
        false);
    field(out, "maxInputTokens", Integer.toString(recipe.maxInputTokens()), false);
    field(out, "truncation", quote(recipe.truncation().name()), false);

    out.append("  \"extra\": {");
    Map<String, String> sorted = new TreeMap<>(recipe.extra());
    boolean first = true;
    for (Map.Entry<String, String> entry : sorted.entrySet()) {
      out.append(first ? "\n" : ",\n")
          .append("    ")
          .append(quote(entry.getKey()))
          .append(": ")
          .append(quote(entry.getValue()));
      first = false;
    }
    out.append(sorted.isEmpty() ? "}\n" : "\n  }\n").append("}\n");
    return out.toString();
  }

  @Override
  public EmbeddingRecipe decode(String text) throws IOException {
    try {
      return new EmbeddingRecipe(
          string(text, "modelId"),
          string(text, "modelVersion"),
          optional(text, "modelDigest"),
          integer(text, "dimension"),
          SimilarityFunction.valueOf(string(text, "metric")),
          Boolean.parseBoolean(raw(text, "normalized")),
          EmbeddingRecipe.Pooling.valueOf(string(text, "pooling")),
          optional(text, "documentPrefix"),
          optional(text, "queryPrefix"),
          integer(text, "maxInputTokens"),
          EmbeddingRecipe.Truncation.valueOf(string(text, "truncation")),
          extra(text));
    } catch (RuntimeException malformed) {
      throw new IOException("malformed recipe document: " + malformed.getMessage(), malformed);
    }
  }

  @Override
  public int schemaVersionOf(String text) throws IOException {
    try {
      return integer(text, "schemaVersion");
    } catch (RuntimeException malformed) {
      throw new IOException("recipe document has no readable schemaVersion", malformed);
    }
  }

  private static void field(StringBuilder out, String key, String value, boolean last) {
    out.append("  ").append(quote(key)).append(": ").append(value).append(last ? "\n" : ",\n");
  }

  /**
   * Parses the {@code extra} object.
   *
   * <p>Quote-aware by necessity: an earlier version split the block on commas, which silently
   * corrupted any value that contained one — and values are arbitrary user strings. Splitting
   * structured text on a delimiter that can appear inside it is the classic version of this bug,
   * and the codec contract test carries a fixture with a comma, a quote, a backslash and a newline
   * in one value precisely so it cannot come back.
   */
  private static Map<String, String> extra(String text) {
    Map<String, String> extra = new LinkedHashMap<>();
    int start = text.indexOf("\"extra\": {");
    if (start < 0) {
      return extra;
    }
    int at = start + "\"extra\": {".length();
    while (at < text.length()) {
      int keyOpen = text.indexOf('"', at);
      if (keyOpen < 0) {
        break;
      }
      // A closing brace before the next quote means the object ended.
      int brace = text.indexOf('}', at);
      if (brace >= 0 && brace < keyOpen) {
        break;
      }
      int keyClose = closingQuote(text, keyOpen);
      if (keyClose < 0) {
        break;
      }
      int colon = text.indexOf(':', keyClose);
      if (colon < 0) {
        break;
      }
      int valueOpen = text.indexOf('"', colon);
      if (valueOpen < 0) {
        break;
      }
      int valueClose = closingQuote(text, valueOpen);
      if (valueClose < 0) {
        break;
      }
      extra.put(
          unquote(text.substring(keyOpen, keyClose + 1)),
          unquote(text.substring(valueOpen, valueClose + 1)));
      at = valueClose + 1;
    }
    return extra;
  }

  /** Index of the quote closing the string opened at {@code open}, honouring backslash escapes. */
  private static int closingQuote(String text, int open) {
    for (int index = open + 1; index < text.length(); index++) {
      char c = text.charAt(index);
      if (c == '\\') {
        index++;
      } else if (c == '"') {
        return index;
      }
    }
    return -1;
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
    StringBuilder out = new StringBuilder(trimmed.length());
    for (int index = 0; index < trimmed.length(); index++) {
      char c = trimmed.charAt(index);
      if (c != '\\' || index + 1 >= trimmed.length()) {
        out.append(c);
        continue;
      }
      char next = trimmed.charAt(++index);
      switch (next) {
        case 'n' -> out.append('\n');
        case 'r' -> out.append('\r');
        case 't' -> out.append('\t');
        case 'u' -> {
          out.append((char) Integer.parseInt(trimmed.substring(index + 1, index + 5), 16));
          index += 4;
        }
        default -> out.append(next);
      }
    }
    return out.toString();
  }

  /**
   * The raw text of a field's value.
   *
   * <p>Tolerant of whitespace around the colon. The first version demanded exactly {@code ": "},
   * which meant a document written by a different codec -- or hand-edited by an auditor, which this
   * format invites -- failed to parse for a reason that had nothing to do with its contents.
   */
  private static String raw(String text, String name) {
    String key = '"' + name + '"';
    int at = text.indexOf(key);
    if (at < 0) {
      throw new IllegalArgumentException("missing field " + name);
    }
    int cursor = at + key.length();
    while (cursor < text.length() && Character.isWhitespace(text.charAt(cursor))) {
      cursor++;
    }
    if (cursor >= text.length() || text.charAt(cursor) != ':') {
      throw new IllegalArgumentException("field " + name + " is not followed by a colon");
    }
    cursor++;
    while (cursor < text.length() && Character.isWhitespace(text.charAt(cursor))) {
      cursor++;
    }
    int end = cursor;
    boolean inString = false;
    while (end < text.length()) {
      char c = text.charAt(end);
      if (c == '\\' && inString) {
        end++;
      } else if (c == '"') {
        inString = !inString;
      } else if (!inString && (c == ',' || c == '\n' || c == '}')) {
        break;
      }
      end++;
    }
    return text.substring(cursor, end).trim();
  }

  private static String string(String text, String name) {
    return unquote(raw(text, name));
  }

  private static Optional<String> optional(String text, String name) {
    String value = raw(text, name);
    return "null".equals(value) ? Optional.empty() : Optional.of(unquote(value));
  }

  private static int integer(String text, String name) {
    return Integer.parseInt(raw(text, name));
  }
}
