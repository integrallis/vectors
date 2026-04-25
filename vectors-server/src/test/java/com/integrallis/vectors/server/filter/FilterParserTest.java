/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server.filter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.db.filter.Filter;
import com.integrallis.vectors.server.ObjectMapperHolder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FilterParserTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();

  private static JsonNode j(String src) throws Exception {
    return JSON.readTree(src);
  }

  @Test
  void nullAndEmptyReturnAll() throws Exception {
    assertThat(FilterParser.parse(null)).isInstanceOf(Filter.All.class);
    assertThat(FilterParser.parse(j("null"))).isInstanceOf(Filter.All.class);
    assertThat(FilterParser.parse(j("{}"))).isInstanceOf(Filter.All.class);
  }

  @Test
  void eqString() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"cat\",\"eq\":\"news\"}"));
    assertThat(f).isEqualTo(new Filter.Eq("cat", "news"));
  }

  @Test
  void eqLongAndDouble() throws Exception {
    assertThat(FilterParser.parse(j("{\"field\":\"n\",\"eq\":42}")))
        .isEqualTo(new Filter.Eq("n", 42L));
    assertThat(FilterParser.parse(j("{\"field\":\"n\",\"eq\":3.14}")))
        .isEqualTo(new Filter.Eq("n", 3.14));
  }

  @Test
  void eqBool() throws Exception {
    assertThat(FilterParser.parse(j("{\"field\":\"ok\",\"eq\":true}")))
        .isEqualTo(new Filter.Eq("ok", true));
  }

  @Test
  void neWrapsInNot() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"cat\",\"ne\":\"news\"}"));
    assertThat(f).isInstanceOf(Filter.Not.class);
    assertThat(((Filter.Not) f).child()).isEqualTo(new Filter.Eq("cat", "news"));
  }

  @Test
  void inStrings() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"cat\",\"in\":[\"news\",\"sports\"]}"));
    assertThat(f).isInstanceOf(Filter.In.class);
    assertThat(((Filter.In) f).values()).containsExactly("news", "sports");
  }

  @Test
  void inNumbersMixedIntegralAndReal() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"n\",\"in\":[1,2,3.5]}"));
    assertThat(f).isInstanceOf(Filter.In.class);
    assertThat(((Filter.In) f).values()).containsExactly(1L, 2L, 3.5);
  }

  @Test
  void ninWrapsInNot() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"cat\",\"nin\":[\"a\",\"b\"]}"));
    assertThat(f).isInstanceOf(Filter.Not.class);
    assertThat(((Filter.Not) f).child()).isInstanceOf(Filter.In.class);
  }

  @Test
  void gteAlone() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"n\",\"gte\":5}"));
    assertThat(f).isEqualTo(new Filter.NumericRange("n", 5.0, true, null, false));
  }

  @Test
  void gtAndLteRange() throws Exception {
    Filter f = FilterParser.parse(j("{\"field\":\"n\",\"gt\":0,\"lte\":10}"));
    assertThat(f).isEqualTo(new Filter.NumericRange("n", 0.0, false, 10.0, true));
  }

  @Test
  void andCompound() throws Exception {
    Filter f =
        FilterParser.parse(j("{\"and\":[{\"field\":\"a\",\"eq\":1},{\"field\":\"b\",\"gte\":2}]}"));
    assertThat(f).isInstanceOf(Filter.And.class);
    assertThat(((Filter.And) f).children()).hasSize(2);
  }

  @Test
  void orAndNotNested() throws Exception {
    Filter f =
        FilterParser.parse(
            j("{\"or\":[{\"not\":{\"field\":\"a\",\"eq\":\"x\"}},{\"field\":\"b\",\"lt\":0}]}"));
    assertThat(f).isInstanceOf(Filter.Or.class);
    Filter.Or or = (Filter.Or) f;
    assertThat(or.children().get(0)).isInstanceOf(Filter.Not.class);
  }

  @Test
  void missingFieldIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"eq\":\"x\"}")))
        .isInstanceOf(FilterParseException.class)
        .hasMessageContaining("field");
  }

  @Test
  void unknownOpIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"field\":\"a\",\"like\":\"x\"}")))
        .isInstanceOf(FilterParseException.class);
  }

  @Test
  void emptyInArrayIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"field\":\"a\",\"in\":[]}")))
        .isInstanceOf(FilterParseException.class);
  }

  @Test
  void mixedTypeInArrayIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"field\":\"a\",\"in\":[1,\"x\"]}")))
        .isInstanceOf(FilterParseException.class);
  }

  @Test
  void gteWithNonNumericValueIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"field\":\"n\",\"gte\":\"foo\"}")))
        .isInstanceOf(FilterParseException.class);
  }

  @Test
  void emptyAndArrayIsRejected() throws Exception {
    assertThatThrownBy(() -> FilterParser.parse(j("{\"and\":[]}")))
        .isInstanceOf(FilterParseException.class);
  }
}
