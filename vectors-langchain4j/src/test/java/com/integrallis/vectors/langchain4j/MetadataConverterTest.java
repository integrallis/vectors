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
package com.integrallis.vectors.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.MetadataValue;
import dev.langchain4j.data.document.Metadata;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class MetadataConverterTest {

  @Nested
  @Tag("unit")
  class ToJavaVectors {

    @Test
    void string() {
      Metadata md = Metadata.from(Map.of("k", "hello"));
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result).containsKey("k");
      assertThat(result.get("k")).isInstanceOf(MetadataValue.Str.class);
      assertThat(((MetadataValue.Str) result.get("k")).value()).isEqualTo("hello");
    }

    @Test
    void integer() {
      Metadata md = new Metadata();
      md.put("k", 42);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result.get("k")).isInstanceOf(MetadataValue.Num.class);
      assertThat(((MetadataValue.Num) result.get("k")).value()).isEqualTo(42.0);
    }

    @Test
    void longValue() {
      Metadata md = new Metadata();
      md.put("k", 100L);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result.get("k")).isInstanceOf(MetadataValue.Num.class);
      assertThat(((MetadataValue.Num) result.get("k")).value()).isEqualTo(100.0);
    }

    @Test
    void floatValue() {
      Metadata md = new Metadata();
      md.put("k", 3.14f);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result.get("k")).isInstanceOf(MetadataValue.Num.class);
      assertThat(((MetadataValue.Num) result.get("k")).value())
          .isCloseTo(3.14, org.assertj.core.data.Offset.offset(0.01));
    }

    @Test
    void doubleValue() {
      Metadata md = new Metadata();
      md.put("k", 2.718);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result.get("k")).isInstanceOf(MetadataValue.Num.class);
      assertThat(((MetadataValue.Num) result.get("k")).value()).isEqualTo(2.718);
    }

    @Test
    void uuid() {
      UUID uuid = UUID.fromString("550e8400-e29b-41d4-a716-446655440000");
      Metadata md = new Metadata();
      md.put("k", uuid);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result.get("k")).isInstanceOf(MetadataValue.Str.class);
      assertThat(((MetadataValue.Str) result.get("k")).value())
          .isEqualTo("550e8400-e29b-41d4-a716-446655440000");
    }

    @Test
    void nullMetadataReturnsEmptyMap() {
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(null);
      assertThat(result).isEmpty();
    }

    @Test
    void emptyMetadataReturnsEmptyMap() {
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(new Metadata());
      assertThat(result).isEmpty();
    }

    @Test
    void multipleEntries() {
      Metadata md = new Metadata();
      md.put("name", "test");
      md.put("count", 5);
      md.put("score", 0.95);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(md);

      assertThat(result).hasSize(3);
      assertThat(result.get("name")).isInstanceOf(MetadataValue.Str.class);
      assertThat(result.get("count")).isInstanceOf(MetadataValue.Num.class);
      assertThat(result.get("score")).isInstanceOf(MetadataValue.Num.class);
    }
  }

  @Nested
  @Tag("unit")
  class ToLangChain4j {

    @Test
    void str() {
      Map<String, MetadataValue> jv = Map.of("k", new MetadataValue.Str("world"));
      Metadata result = MetadataConverter.toLangChain4j(jv);

      assertThat(result.getString("k")).isEqualTo("world");
    }

    @Test
    void numIntegral() {
      Map<String, MetadataValue> jv = Map.of("k", new MetadataValue.Num(42.0));
      Metadata result = MetadataConverter.toLangChain4j(jv);

      assertThat(result.getLong("k")).isEqualTo(42L);
    }

    @Test
    void numFractional() {
      Map<String, MetadataValue> jv = Map.of("k", new MetadataValue.Num(3.14));
      Metadata result = MetadataConverter.toLangChain4j(jv);

      assertThat(result.getDouble("k")).isEqualTo(3.14);
    }

    @Test
    void bool() {
      Map<String, MetadataValue> jv = Map.of("k", new MetadataValue.Bool(true));
      Metadata result = MetadataConverter.toLangChain4j(jv);

      // LangChain4j Metadata doesn't support boolean; stored as string
      assertThat(result.getString("k")).isEqualTo("true");
    }

    @Test
    void tags() {
      Map<String, MetadataValue> jv =
          Map.of("k", new MetadataValue.Tags(java.util.List.of("a", "b", "c")));
      Metadata result = MetadataConverter.toLangChain4j(jv);

      // Tags stored as comma-separated string since Metadata doesn't support Lists
      assertThat(result.getString("k")).isEqualTo("a,b,c");
    }

    @Test
    void nullMapReturnsEmptyMetadata() {
      Metadata result = MetadataConverter.toLangChain4j(null);
      assertThat(result.toMap()).isEmpty();
    }

    @Test
    void emptyMapReturnsEmptyMetadata() {
      Metadata result = MetadataConverter.toLangChain4j(Map.of());
      assertThat(result.toMap()).isEmpty();
    }
  }

  @Nested
  @Tag("unit")
  class RoundTrip {

    @Test
    void stringRoundTrip() {
      Metadata original = Metadata.from(Map.of("name", "test"));
      Map<String, MetadataValue> jv = MetadataConverter.toJavaVectors(original);
      Metadata back = MetadataConverter.toLangChain4j(jv);

      assertThat(back.getString("name")).isEqualTo("test");
    }

    @Test
    void numericRoundTrip() {
      Metadata original = new Metadata();
      original.put("count", 42L);
      Map<String, MetadataValue> jv = MetadataConverter.toJavaVectors(original);
      Metadata back = MetadataConverter.toLangChain4j(jv);

      assertThat(back.getLong("count")).isEqualTo(42L);
    }
  }
}
