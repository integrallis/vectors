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

import com.integrallis.vectors.core.filter.Filter;
import com.integrallis.vectors.core.filter.Filters;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CoreValueTypesTest {

  @Test
  void documentFactoriesAndMetadataValuesPreserveTheirContracts() {
    float[] vector = {1f, 2f};
    Document minimal = Document.of("id", vector);
    Document withText = Document.of("text", vector, "body");
    assertThat(minimal.id()).isEqualTo("id");
    assertThat(minimal.metadata()).isEmpty();
    assertThat(withText.text()).isEqualTo("body");

    Map<String, MetadataValue> metadata =
        Map.of(
            "str", new MetadataValue.Str("value"),
            "num", new MetadataValue.Num(3.5),
            "bool", new MetadataValue.Bool(true),
            "tags", new MetadataValue.Tags(List.of("a", "b")));
    Document full = new Document("full", vector, null, metadata);
    assertThat(full.metadata()).isEqualTo(metadata);
    assertThatThrownBy(() -> new Document(null, vector, null, Map.of()))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void filterFactoriesBuildTheFullTypedAst() {
    Filter eqString = Filters.eq("name", "Ada");
    Filter eqLong = Filters.eq("count", 2L);
    Filter eqDouble = Filters.eq("score", 0.5);
    Filter eqBoolean = Filters.eq("active", true);
    Filter gte = Filters.gte("score", 0.1);
    Filter lte = Filters.lte("score", 0.9);
    Filter gt = Filters.gt("score", 0.1);
    Filter lt = Filters.lt("score", 0.9);
    Filter between = Filters.between("score", 0.1, 0.9);
    Filter inString = Filters.inStr("name", "Ada", "Grace");
    Filter inNumber = Filters.inNum("score", 0.1, 0.9);
    Filter all = Filters.all();

    assertThat(Filters.and(eqString, gte)).isInstanceOf(Filter.And.class);
    assertThat(Filters.or(eqLong, lte)).isInstanceOf(Filter.Or.class);
    assertThat(Filters.not(eqBoolean)).isInstanceOf(Filter.Not.class);
    assertThat(List.of(eqDouble, gt, lt, between, inString, inNumber, all))
        .allMatch(Filter.class::isInstance);
  }

  @Test
  void filterNodesRejectInvalidStructureAndCopyLists() {
    assertThatThrownBy(() -> new Filter.Eq(null, "x")).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.Eq("x", null)).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.NumericRange(null, null, false, null, false))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.In(null, List.of()))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.And(null)).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.And(List.of()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Filter.Or(null)).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new Filter.Or(List.of())).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Filter.Not(null)).isInstanceOf(NullPointerException.class);

    Filter child = Filters.all();
    assertThatThrownBy(() -> new Filter.And(List.of(child)).children().add(child))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> new Filter.In("x", List.of("a")).values().add("b"))
        .isInstanceOf(UnsupportedOperationException.class);
  }
}
