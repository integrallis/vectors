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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class BuiltinRecipeCodecTest {
  private final BuiltinRecipeCodec codec = new BuiltinRecipeCodec();

  private static EmbeddingRecipe recipe(Map<String, String> extra) {
    return new EmbeddingRecipe(
        "model \"quoted\"",
        "revision\\one",
        Optional.of("a".repeat(64)),
        384,
        SimilarityFunction.COSINE,
        false,
        EmbeddingRecipe.Pooling.CLS,
        Optional.of("document:\t"),
        Optional.of("query:\r\n"),
        512,
        EmbeddingRecipe.Truncation.TAIL,
        extra);
  }

  @Test
  void allFieldsAndControlCharactersSurviveRoundTrip() throws IOException {
    StringBuilder controls = new StringBuilder("commas, braces{} quote\" slash\\ λ😀");
    for (int c = 0; c < 32; c++) controls.append((char) c);
    EmbeddingRecipe original = recipe(Map.of("escaped\"key", controls.toString(), "empty", ""));
    String encoded = codec.encode(original);
    assertThat(codec.decode(encoded)).isEqualTo(original);
    assertThat(codec.decode(encoded).recipeHash()).isEqualTo(original.recipeHash());
    assertThat(encoded).doesNotContain("\r").contains("\\u0000", "\\u001f");
  }

  @Test
  void absentValuesAndEmptyMapRemainAbsentRatherThanLiteralNull() throws IOException {
    EmbeddingRecipe original =
        EmbeddingRecipe.declared("remote", "latest", 16, SimilarityFunction.DOT_PRODUCT);
    String encoded = codec.encode(original);
    assertThat(encoded)
        .contains("\"modelDigest\": null", "\"documentPrefix\": null", "\"queryPrefix\": null");
    assertThat(codec.decode(encoded)).isEqualTo(original);
    assertThat(codec.schemaVersionOf(encoded)).isEqualTo(codec.schemaVersion());
  }

  @Test
  void mapInsertionOrderCannotChangeEncodedBytes() {
    Map<String, String> a = new LinkedHashMap<>(), b = new LinkedHashMap<>();
    a.put("z", "last");
    a.put("a", "first");
    b.put("a", "first");
    b.put("z", "last");
    assertThat(codec.encode(recipe(a))).isEqualTo(codec.encode(recipe(b)));
    assertThat(codec.encode(recipe(a)).indexOf("\"a\": "))
        .isLessThan(codec.encode(recipe(a)).indexOf("\"z\": "));
  }

  @Test
  void unknownFieldsAndEditedAuditHashDoNotOverrideRecipeIdentity() throws IOException {
    EmbeddingRecipe original = recipe(Map.of("tokenizer", "custom"));
    String encoded = codec.encode(original).replace(original.recipeHash(), "not-the-recipe-hash");
    encoded = encoded.substring(0, 1) + "\n  \"future-field\": true," + encoded.substring(1);
    assertThat(codec.decode(encoded)).isEqualTo(original);
    assertThat(codec.decode(encoded).recipeHash()).isEqualTo(original.recipeHash());
  }

  @Test
  void schemaVersionCanBeReadWithoutUnderstandingOtherFields() throws IOException {
    assertThat(codec.schemaVersionOf("{ \"schemaVersion\" \t:\n 42, \"future\": [] }"))
        .isEqualTo(42);
    assertThat(codec.schemaVersionOf("{\"schemaVersion\":1}")).isEqualTo(1);
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "{}",
        "{\"schemaVersion\" 1}",
        "{\"schemaVersion\":\"bad\"}",
        "{\"schemaVersion\":2147483648}"
      })
  void malformedSchemaIsReportedAsIoFailure(String document) {
    assertThatThrownBy(() -> codec.schemaVersionOf(document))
        .isInstanceOf(IOException.class)
        .hasCauseInstanceOf(RuntimeException.class);
  }

  @Test
  void missingAndInvalidRequiredFieldsAreRejected() {
    String valid = codec.encode(recipe(Map.of()));
    for (String malformed :
        new String[] {
          "{}",
          valid.replace("\"dimension\": 384", "\"dimension\": 0"),
          valid.replace("\"metric\": \"COSINE\"", "\"metric\": \"unknown\""),
          valid.replace("\"modelId\":", "\"modelId\" ")
        }) {
      assertThatThrownBy(() -> codec.decode(malformed)).isInstanceOf(IOException.class);
    }
  }
}
