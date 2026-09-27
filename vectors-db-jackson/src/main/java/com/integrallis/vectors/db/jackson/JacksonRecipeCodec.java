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
package com.integrallis.vectors.db.jackson;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.RecipeCodec;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;

/**
 * A {@link RecipeCodec} backed by Jackson, for applications already standardised on it.
 *
 * <p>Opt-in: put this module on the classpath and {@link
 * com.integrallis.vectors.core.RecipeCodecs#discover()} finds it, or name it in {@code
 * -Dvectors.recipeCodec}. The core library keeps no JSON dependency of its own.
 *
 * <p>Interchangeable with the built-in codec on existing data, because {@link
 * EmbeddingRecipe#recipeHash()} is computed from a canonical field rendering rather than the
 * serialised text — so adding or removing this module cannot invalidate a stored collection.
 *
 * <p>Determinism, which the interface requires, needs three explicit settings here. Jackson
 * provides none of them by default: {@code ORDER_MAP_ENTRIES_BY_KEYS} (the {@code extra} map would
 * otherwise follow iteration order), an indenter pinned to {@code "\n"} (the default pretty printer
 * uses {@code SYSTEM_LINEFEED}, so the same recipe would write CRLF on Windows), and nulls written
 * rather than omitted.
 */
public final class JacksonRecipeCodec implements RecipeCodec {

  /** Schema version of the layout this codec reads and writes. Matches the built-in codec's. */
  public static final int SCHEMA_VERSION = 1;

  private static final ObjectMapper MAPPER =
      JsonMapper.builder()
          .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
          .enable(SerializationFeature.INDENT_OUTPUT)
          .build();

  private static final DefaultPrettyPrinter PRINTER =
      new DefaultPrettyPrinter()
          .withObjectIndenter(new DefaultIndenter("  ", "\n"))
          .withArrayIndenter(new DefaultIndenter("  ", "\n"));

  /** The on-disk shape, decoupled from the domain record so the two can evolve independently. */
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

  @Override
  public int schemaVersion() {
    return SCHEMA_VERSION;
  }

  @Override
  public String encode(EmbeddingRecipe recipe) throws IOException {
    RecipeDocument document =
        new RecipeDocument(
            SCHEMA_VERSION,
            // An auditor's convenience, never trusted on read.
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
    try {
      return MAPPER.writer(PRINTER).writeValueAsString(document) + "\n";
    } catch (JsonProcessingException failure) {
      throw new IOException("failed to encode recipe: " + failure.getMessage(), failure);
    }
  }

  @Override
  public EmbeddingRecipe decode(String text) throws IOException {
    RecipeDocument document = read(text);
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

  @Override
  public int schemaVersionOf(String text) throws IOException {
    return read(text).schemaVersion();
  }

  private static RecipeDocument read(String text) throws IOException {
    try {
      // Unknown properties tolerated: a newer writer may have added fields, and failing here would
      // turn a forward-compatible file into an unopenable collection.
      return MAPPER
          .readerFor(RecipeDocument.class)
          .without(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
          .readValue(text);
    } catch (RuntimeException | JsonProcessingException malformed) {
      throw new IOException("malformed recipe document: " + malformed.getMessage(), malformed);
    }
  }
}
