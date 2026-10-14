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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
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
      Map<String, Object> document = new JsonReader(text).document();
      return new EmbeddingRecipe(
          string(document, "modelId"),
          string(document, "modelVersion"),
          optional(document, "modelDigest"),
          integer(document, "dimension"),
          SimilarityFunction.valueOf(string(document, "metric")),
          bool(document, "normalized"),
          EmbeddingRecipe.Pooling.valueOf(string(document, "pooling")),
          optional(document, "documentPrefix"),
          optional(document, "queryPrefix"),
          integer(document, "maxInputTokens"),
          EmbeddingRecipe.Truncation.valueOf(string(document, "truncation")),
          extra(document));
    } catch (RuntimeException malformed) {
      throw new IOException("malformed recipe document: " + malformed.getMessage(), malformed);
    }
  }

  @Override
  public int schemaVersionOf(String text) throws IOException {
    try {
      return integer(new JsonReader(text).document(), "schemaVersion");
    } catch (RuntimeException malformed) {
      throw new IOException("recipe document has no readable schemaVersion", malformed);
    }
  }

  private static void field(StringBuilder out, String key, String value, boolean last) {
    out.append("  ").append(quote(key)).append(": ").append(value).append(last ? "\n" : ",\n");
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

  private static String string(Map<String, Object> document, String name) {
    Object value = document.get(name);
    if (value instanceof String string) return string;
    throw new IllegalArgumentException("field " + name + " must be a string");
  }

  private static Optional<String> optional(Map<String, Object> document, String name) {
    return document.get(name) == null ? Optional.empty() : Optional.of(string(document, name));
  }

  private static boolean bool(Map<String, Object> document, String name) {
    Object value = document.get(name);
    if (value instanceof Boolean bool) return bool;
    throw new IllegalArgumentException("field " + name + " must be a boolean");
  }

  private static int integer(Map<String, Object> document, String name) {
    Object value = document.get(name);
    if (value instanceof JsonNumber number) return Integer.parseInt(number.text());
    throw new IllegalArgumentException("field " + name + " must be an integer");
  }

  private static Map<String, String> extra(Map<String, Object> document) {
    Object value = document.get("extra");
    if (value == null) return Map.of();
    if (!(value instanceof Map<?, ?> fields)) {
      throw new IllegalArgumentException("field extra must be an object");
    }
    Map<String, String> extra = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : fields.entrySet()) {
      if (!(entry.getValue() instanceof String string)) {
        throw new IllegalArgumentException("extra values must be strings");
      }
      // JsonReader only creates objects with string keys.
      extra.put((String) entry.getKey(), string);
    }
    return extra;
  }

  /** Retains number spelling so schema integers cannot silently truncate fractions or overflow. */
  private record JsonNumber(String text) {}

  /**
   * Structural JSON reader for a small recipe document, including unknown future values. Duplicate
   * keys are rejected rather than choosing an ambiguous recipe identity. The nesting bound prevents
   * an unknown field from exhausting the Java stack; no external JSON dependency is required.
   */
  private static final class JsonReader {
    private static final int MAX_DEPTH = 128;
    private final String text;
    private int at;

    private JsonReader(String text) {
      this.text = java.util.Objects.requireNonNull(text, "text");
    }

    private Map<String, Object> document() {
      whitespace();
      Map<String, Object> result = object(0);
      whitespace();
      if (at != text.length()) throw error("trailing content");
      return result;
    }

    private Map<String, Object> object(int depth) {
      require('{');
      Map<String, Object> result = new LinkedHashMap<>();
      whitespace();
      if (take('}')) return result;
      do {
        whitespace();
        String key = string();
        whitespace();
        require(':');
        if (result.containsKey(key)) throw error("duplicate field " + key);
        result.put(key, value(depth + 1));
        whitespace();
        if (take('}')) return result;
        require(',');
      } while (true);
    }

    private List<Object> array(int depth) {
      require('[');
      List<Object> result = new ArrayList<>();
      whitespace();
      if (take(']')) return result;
      do {
        result.add(value(depth + 1));
        whitespace();
        if (take(']')) return result;
        require(',');
      } while (true);
    }

    private Object value(int depth) {
      if (depth > MAX_DEPTH) throw error("JSON nesting exceeds " + MAX_DEPTH);
      whitespace();
      if (at == text.length()) throw error("missing value");
      return switch (text.charAt(at)) {
        case '{' -> object(depth);
        case '[' -> array(depth);
        case '"' -> string();
        case 't' -> literal("true", Boolean.TRUE);
        case 'f' -> literal("false", Boolean.FALSE);
        case 'n' -> literal("null", null);
        default -> number();
      };
    }

    private Object literal(String token, Object value) {
      if (!text.startsWith(token, at)) throw error("invalid literal");
      at += token.length();
      return value;
    }

    private JsonNumber number() {
      int start = at;
      take('-');
      if (!take('0')) digits();
      if (take('.')) digits();
      if (take('e') || take('E')) {
        if (!take('+')) take('-');
        digits();
      }
      return new JsonNumber(text.substring(start, at));
    }

    private void digits() {
      int start = at;
      while (at < text.length() && text.charAt(at) >= '0' && text.charAt(at) <= '9') at++;
      if (start == at) throw error("expected digit");
    }

    private String string() {
      require('"');
      StringBuilder result = new StringBuilder();
      while (at < text.length()) {
        char c = text.charAt(at++);
        if (c == '"') return result.toString();
        if (c < 0x20) throw error("unescaped control character");
        if (c != '\\') {
          result.append(c);
          continue;
        }
        if (at == text.length()) throw error("incomplete string escape");
        switch (text.charAt(at++)) {
          case '"' -> result.append('"');
          case '\\' -> result.append('\\');
          case '/' -> result.append('/');
          case 'b' -> result.append('\b');
          case 'f' -> result.append('\f');
          case 'n' -> result.append('\n');
          case 'r' -> result.append('\r');
          case 't' -> result.append('\t');
          case 'u' -> result.append(unicodeEscape());
          default -> throw error("invalid string escape");
        }
      }
      throw error("unterminated string");
    }

    private char unicodeEscape() {
      int value = 0;
      for (int i = 0; i < 4; i++) {
        if (at == text.length()) throw error("incomplete unicode escape");
        char c = text.charAt(at++);
        int digit =
            c >= '0' && c <= '9'
                ? c - '0'
                : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (digit < 0) throw error("invalid unicode escape");
        value = value * 16 + digit;
      }
      return (char) value;
    }

    private void whitespace() {
      while (at < text.length()) {
        char c = text.charAt(at);
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        at++;
      }
    }

    private boolean take(char c) {
      if (at < text.length() && text.charAt(at) == c) {
        at++;
        return true;
      }
      return false;
    }

    private void require(char c) {
      if (!take(c)) throw error("expected " + c);
    }

    private IllegalArgumentException error(String message) {
      return new IllegalArgumentException(message + " at character " + at);
    }
  }
}
