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

import com.integrallis.vectors.core.BuiltinRecipeCodec;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.RecipeCodec;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The four obligations {@link RecipeCodec} documents, run against every codec.
 *
 * <p>Written as a contract rather than per-implementation tests so that a third codec — an
 * application's own — can be dropped into {@link #codecs()} and held to the same bar.
 */
class RecipeCodecContractTest {

  static Stream<RecipeCodec> codecs() {
    // The Jackson codec lives in a module vectors-db does not depend on, by design, so only the
    // built-in is exercised here; vectors-db-jackson runs this same contract against its own.
    return Stream.of(new BuiltinRecipeCodec());
  }

  private static EmbeddingRecipe full() {
    return new EmbeddingRecipe(
        "nomic-embed-text",
        "1.5",
        Optional.of("a".repeat(64)),
        768,
        SimilarityFunction.COSINE,
        true,
        EmbeddingRecipe.Pooling.MEAN,
        Optional.of("search_document: "),
        Optional.of("search_query: "),
        8192,
        EmbeddingRecipe.Truncation.REJECT,
        Map.of("tokenizer", "bert-base", "note", "quoted \"value\", a \\ and a \n newline"));
  }

  private static EmbeddingRecipe minimal() {
    return EmbeddingRecipe.declared(
        "text-embedding-3-small", "2024-01", 1536, SimilarityFunction.COSINE);
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("round-trip fidelity, including the awkward values")
  void roundTripsEveryField(RecipeCodec codec) throws IOException {
    for (EmbeddingRecipe original : new EmbeddingRecipe[] {full(), minimal()}) {
      EmbeddingRecipe decoded = codec.decode(codec.encode(original));
      assertEquals(original, decoded, "a field was dropped or mangled by " + codec.getClass());
      // The hash must survive too, or the manifest anchor stops matching after a rewrite.
      assertEquals(original.recipeHash(), decoded.recipeHash());
    }
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("determinism: same recipe, same bytes")
  void encodingIsDeterministic(RecipeCodec codec) throws IOException {
    assertEquals(codec.encode(full()), codec.encode(full()));
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("determinism: map insertion order must not leak into the bytes")
  void mapOrderDoesNotLeak(RecipeCodec codec) throws IOException {
    Map<String, String> forward = new LinkedHashMap<>();
    forward.put("alpha", "1");
    forward.put("beta", "2");
    Map<String, String> reverse = new LinkedHashMap<>();
    reverse.put("beta", "2");
    reverse.put("alpha", "1");
    assertEquals(
        codec.encode(withExtra(forward)),
        codec.encode(withExtra(reverse)),
        codec.getClass() + " leaks map iteration order into the file");
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("determinism: no platform line separator")
  void noCarriageReturns(RecipeCodec codec) throws IOException {
    assertFalse(
        codec.encode(full()).contains("\r"),
        codec.getClass() + " uses a platform-dependent line separator");
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("an unknown future field does not break decoding")
  void unknownFieldsTolerated(RecipeCodec codec) throws IOException {
    EmbeddingRecipe original = full();
    String text =
        codec
            .encode(original)
            .replaceFirst("(\\n\\s*)\"modelId\"", "$1\"fieldFromTheFuture\": 42,$1\"modelId\"");
    assertTrue(text.contains("fieldFromTheFuture"), "the edit must actually land");
    assertEquals(
        original.modelId(),
        codec.decode(text).modelId(),
        codec.getClass() + " fails on an unknown field, which would make a newer file unopenable");
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("every field is written, so an absent key never means an unset one")
  void nullsAreWrittenNotOmitted(RecipeCodec codec) throws IOException {
    // Carried over from the codec-specific tests this contract replaced. A minimal recipe has no
    // digest and no prefixes; those keys must still appear. Omitting them would make "this writer
    // had
    // no digest" indistinguishable from "this writer did not know about digests", and that
    // difference
    // is exactly the attested/declared line.
    String text = codec.encode(minimal());
    for (String key : new String[] {"modelDigest", "documentPrefix", "queryPrefix"}) {
      assertTrue(text.contains(key), codec.getClass() + " omitted " + key + " when absent");
    }
    // And the round trip must restore them as absent rather than as the string "null".
    EmbeddingRecipe decoded = codec.decode(text);
    assertTrue(decoded.modelDigest().isEmpty(), "modelDigest must decode back to empty");
    assertTrue(decoded.documentPrefix().isEmpty(), "documentPrefix must decode back to empty");
  }

  @ParameterizedTest
  @MethodSource("codecs")
  @DisplayName("schemaVersionOf works without fully interpreting the document")
  void schemaVersionIsReadableAlone(RecipeCodec codec) throws IOException {
    assertEquals(codec.schemaVersion(), codec.schemaVersionOf(codec.encode(full())));
  }

  private static EmbeddingRecipe withExtra(Map<String, String> extra) {
    EmbeddingRecipe base = full();
    return new EmbeddingRecipe(
        base.modelId(),
        base.modelVersion(),
        base.modelDigest(),
        base.dimension(),
        base.metric(),
        base.normalized(),
        base.pooling(),
        base.documentPrefix(),
        base.queryPrefix(),
        base.maxInputTokens(),
        base.truncation(),
        extra);
  }
}
