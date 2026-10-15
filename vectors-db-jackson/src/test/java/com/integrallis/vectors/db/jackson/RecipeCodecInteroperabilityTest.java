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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.vectors.core.BuiltinRecipeCodec;
import com.integrallis.vectors.core.EmbeddingRecipe;
import com.integrallis.vectors.core.SimilarityFunction;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class RecipeCodecInteroperabilityTest {
  private final BuiltinRecipeCodec builtin = new BuiltinRecipeCodec();
  private final JacksonRecipeCodec jackson = new JacksonRecipeCodec();
  private final ObjectMapper mapper = JsonMapper.builder().build();

  @ParameterizedTest
  @ValueSource(ints = {1, 29, 711, 9043})
  void bothCodecsReadCompactPrettyReorderedAndEscapedJsonWithIdenticalIdentity(int seed)
      throws IOException {
    Random random = new Random(seed);
    for (int sample = 0; sample < 25; sample++) {
      Map<String, String> extra = new LinkedHashMap<>();
      extra.put("modelId", "nested distractor");
      extra.put("schemaVersion", "999");
      extra.put("escaped\"\\/", randomString(random));
      EmbeddingRecipe recipe =
          new EmbeddingRecipe(
              "model-" + randomString(random),
              randomString(random),
              random.nextBoolean() ? Optional.of("ab".repeat(32)) : Optional.empty(),
              1 + random.nextInt(4096),
              SimilarityFunction.values()[random.nextInt(SimilarityFunction.values().length)],
              random.nextBoolean(),
              EmbeddingRecipe.Pooling.values()[
                  random.nextInt(EmbeddingRecipe.Pooling.values().length)],
              random.nextBoolean() ? Optional.of(randomString(random)) : Optional.empty(),
              random.nextBoolean() ? Optional.of(randomString(random)) : Optional.empty(),
              1 + random.nextInt(32768),
              EmbeddingRecipe.Truncation.values()[
                  random.nextInt(EmbeddingRecipe.Truncation.values().length)],
              extra);
      ObjectNode original = (ObjectNode) mapper.readTree(jackson.encode(recipe));
      ObjectNode reordered = mapper.createObjectNode();
      reordered.set(
          "future",
          mapper.readTree("{\"dimension\":0,\"schemaVersion\":999,\"extra\":[null,false,1e300]}"));
      reordered.set("extra", original.get("extra"));
      original.properties().forEach(field -> reordered.set(field.getKey(), field.getValue()));
      for (String document :
          new String[] {
            builtin.encode(recipe),
            jackson.encode(recipe),
            mapper.writeValueAsString(reordered),
            mapper.writerWithDefaultPrettyPrinter().writeValueAsString(reordered)
          }) {
        assertThat(builtin.decode(document)).isEqualTo(recipe);
        assertThat(jackson.decode(document)).isEqualTo(recipe);
        assertThat(builtin.decode(document).recipeHash())
            .isEqualTo(jackson.decode(document).recipeHash());
        assertThat(builtin.schemaVersionOf(document)).isEqualTo(jackson.schemaVersionOf(document));
      }
    }
  }

  @Test
  void duplicatePolicyMatchesStrictJacksonWithoutChangingItsOptionalModuleDefault() {
    ObjectMapper strict =
        JsonMapper.builder().enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION).build();
    for (String document :
        new String[] {
          "{\"schemaVersion\":1,\"schemaVersion\":2}",
          "{\"schemaVersion\":1,\"future\":{\"a\":null,\"a\":false}}",
          "{\"schemaVersion\":1,\"schema\\u0056ersion\":1}"
        }) {
      assertThatThrownBy(() -> strict.readTree(document)).isInstanceOf(IOException.class);
      assertThatThrownBy(() -> builtin.schemaVersionOf(document)).isInstanceOf(IOException.class);
    }
  }

  @Test
  void malformedStringsAndNumbersAreRejectedByBothStructuralReaders() {
    for (String value :
        new String[] {
          "01", "1.", "1e", "+1", "NaN", "\"bad\\q\"", "\"\\u12G4\"", "\"raw\nline\"", "[1,]"
        }) {
      String document = "{\"schemaVersion\":1,\"future\":" + value + "}";
      assertThatThrownBy(() -> mapper.readTree(document)).isInstanceOf(IOException.class);
      assertThatThrownBy(() -> builtin.schemaVersionOf(document)).isInstanceOf(IOException.class);
    }
  }

  private static String randomString(Random random) {
    String alphabet = " ab,{}[]:\"\\/\b\f\n\r\tλ😀";
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < 40; i++) text.append(alphabet.charAt(random.nextInt(alphabet.length())));
    return text.toString();
  }
}
