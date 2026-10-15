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
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class BuiltinRecipeCodecStructureTest {
  private final BuiltinRecipeCodec codec = new BuiltinRecipeCodec();
  private final EmbeddingRecipe recipe =
      new EmbeddingRecipe(
          "model",
          "v1",
          Optional.empty(),
          16,
          SimilarityFunction.COSINE,
          true,
          EmbeddingRecipe.Pooling.MEAN,
          Optional.empty(),
          Optional.of("query:\n"),
          512,
          EmbeddingRecipe.Truncation.HEAD,
          Map.of(
              "modelId", "nested-model", "tokenizer", "comma,brace}colon:quote\"", "slash", "\\/"));

  @ParameterizedTest
  @ValueSource(strings = {"", " ", "\t", "\n", "\r\n", " \t\r\n "})
  void whitespaceCannotDropMetadataOrChangeIdentity(String space) throws IOException {
    String text =
        codec.encode(recipe).replace("\"extra\": {", "\"extra\"" + space + ":" + space + "{");
    assertThat(codec.decode(text)).isEqualTo(recipe);
    assertThat(codec.decode(text).recipeHash()).isEqualTo(recipe.recipeHash());
  }

  @Test
  void nestedFieldsBeforeRealFieldsCannotShadowRecipeOrSchema() throws IOException {
    String json = codec.encode(recipe);
    String unknown =
        "\"future\": {\"modelId\":\"wrong\",\"dimension\":1,\"schemaVersion\":999,"
            + "\"extra\":{\"wrong\":\"wrong\"},\"list\":[null,true,false,-12.5e+4,{\"x\":[1,2]}]},";
    json = "{" + unknown + json.substring(1);
    assertThat(codec.decode(json)).isEqualTo(recipe);
    assertThat(codec.schemaVersionOf(json)).isEqualTo(1);
  }

  @Test
  void allJsonEscapesAndEscapedFieldNamesAreDecoded() throws IOException {
    String json =
        codec
            .encode(recipe)
            .replace(
                "\"modelId\": \"model\"",
                "\"model\\u0049d\": \"\\b\\f\\n\\r\\t\\/\\\\\\\"\\u03bb\\uD83D\\uDE00\"");
    assertThat(codec.decode(json).modelId()).isEqualTo("\b\f\n\r\t/\\\"λ😀");
    assertThat(codec.decode(json).extra()).isEqualTo(recipe.extra());
  }

  @Test
  void schemaInspectionDoesNotInterpretUnknownFutureFields() throws IOException {
    assertThat(codec.schemaVersionOf("{\"future\":[[{},1e300,null],false],\"schemaVersion\":42}"))
        .isEqualTo(42);
  }

  @Test
  void absentOrNullOptionalMetadataHasAnExplicitEmptyMeaning() throws IOException {
    String json = codec.encode(recipe);
    assertThat(codec.decode(json.replace("  \"modelDigest\": null,\n", "")).modelDigest())
        .isEmpty();
    assertThat(
            codec
                .decode(json.substring(0, json.indexOf("  \"extra\"")) + "\"extra\":null}")
                .extra())
        .isEmpty();
    assertThat(
            codec
                .decode(json.substring(0, json.indexOf("  \"extra\"")) + "\"future\":null}")
                .extra())
        .isEmpty();
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "\"modelId\":\"duplicate\"",
        "\"schemaVersion\":1",
        "\"model\\u0049d\":\"duplicate\""
      })
  void duplicateTopLevelKeysAreRejectedEvenWhenEscaped(String field) {
    String json = "{" + field + "," + codec.encode(recipe).substring(1);
    assertThatThrownBy(() -> codec.decode(json))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("duplicate field");
    assertThatThrownBy(() -> codec.schemaVersionOf(json)).isInstanceOf(IOException.class);
  }

  @Test
  void duplicateExtraKeysAndUnknownObjectKeysAreRejectedIncludingNullFirstValues() {
    for (String body :
        new String[] {
          "\"extra\":{\"a\":\"1\",\"a\":\"2\"}", "\"future\":{\"a\":null,\"a\":true}"
        }) {
      String json = "{" + body + "," + codec.encode(recipe).substring(1);
      assertThatThrownBy(() -> codec.decode(json))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("duplicate field");
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"null", "true", "[]", "{}", "\"16\"", "16.0", "16e0", "2147483648"})
  void dimensionRequiresAnIntWithoutCoercionOrTruncation(String value) {
    String json = codec.encode(recipe).replace("\"dimension\": 16", "\"dimension\": " + value);
    assertThatThrownBy(() -> codec.decode(json)).isInstanceOf(IOException.class);
  }

  @ParameterizedTest
  @ValueSource(strings = {"null", "\"true\"", "1", "{}", "[]"})
  void normalizedRequiresABoolean(String value) {
    String json = codec.encode(recipe).replace("\"normalized\": true", "\"normalized\": " + value);
    assertThatThrownBy(() -> codec.decode(json)).isInstanceOf(IOException.class);
  }

  @ParameterizedTest
  @ValueSource(strings = {"[]", "\"text\"", "1", "{\"a\":null}", "{\"a\":false}", "{\"a\":{}}"})
  void extraRequiresAnObjectOfStrings(String value) {
    String json = codec.encode(recipe);
    String malformed = json.substring(0, json.indexOf("  \"extra\"")) + "\"extra\":" + value + "}";
    assertThatThrownBy(() -> codec.decode(malformed)).isInstanceOf(IOException.class);
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "",
        "[]",
        "null",
        "{} trailing",
        "{\"schemaVersion\":1,}",
        "{\"schemaVersion\":1",
        "{schemaVersion:1}",
        "{\"schemaVersion\":+1}",
        "{\"schemaVersion\":01}",
        "{\"schemaVersion\":.1}",
        "{\"schemaVersion\":1.}",
        "{\"schemaVersion\":1e}",
        "{\"schemaVersion\":1e+}",
        "{\"schemaVersion\":1e--2}",
        "{\"schemaVersion\":NaN}",
        "{\"schemaVersion\":Infinity}",
        "{\"schemaVersion\":1,\"future\":[1,]}",
        "{\"schemaVersion\":1,\"future\":tru}",
        "{\"schemaVersion\":1,\"future\":\"\\q\"}",
        "{\"schemaVersion\":1,\"future\":\"\\u12G4\"}",
        "{\"schemaVersion\":1,\"future\":\"\\u1\"}",
        "{\"schemaVersion\":1,\"future\":\"line\nbreak\"}",
        "{\"schemaVersion\":1}\u00a0"
      })
  void invalidJsonIsRejectedInsteadOfPartiallyRead(String json) {
    assertThatThrownBy(() -> codec.schemaVersionOf(json)).isInstanceOf(IOException.class);
  }

  @Test
  void deeplyNestedUnknownDataFailsAsIoErrorRatherThanStackOverflow() {
    String json =
        "{\"schemaVersion\":1,\"future\":" + "[".repeat(129) + "0" + "]".repeat(129) + "}";
    assertThatThrownBy(() -> codec.schemaVersionOf(json))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("schemaVersion");
  }
}
