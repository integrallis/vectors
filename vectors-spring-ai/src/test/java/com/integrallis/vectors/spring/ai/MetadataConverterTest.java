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
package com.integrallis.vectors.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.db.MetadataValue;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class MetadataConverterTest {

  @Nested
  @Tag("unit")
  class ToJavaVectors {

    @Test
    void stringValue() {
      Map<String, Object> springMd = Map.of("key", "hello");
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(result).containsKey("key");
      assertThat(result.get("key")).isInstanceOf(MetadataValue.Str.class);
      assertThat(((MetadataValue.Str) result.get("key")).value()).isEqualTo("hello");
    }

    @Test
    void integerValue() {
      Map<String, Object> springMd = Map.of("count", 42);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(result.get("count")).isInstanceOf(MetadataValue.Num.class);
      assertThat(((MetadataValue.Num) result.get("count")).value()).isEqualTo(42.0);
    }

    @Test
    void longValue() {
      Map<String, Object> springMd = Map.of("big", 123456789L);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(((MetadataValue.Num) result.get("big")).value()).isEqualTo(123456789.0);
    }

    @Test
    void doubleValue() {
      Map<String, Object> springMd = Map.of("pi", 3.14);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(((MetadataValue.Num) result.get("pi")).value()).isEqualTo(3.14);
    }

    @Test
    void floatValue() {
      Map<String, Object> springMd = Map.of("f", 2.5f);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(((MetadataValue.Num) result.get("f")).value()).isEqualTo(2.5);
    }

    @Test
    void booleanValue() {
      Map<String, Object> springMd = Map.of("active", true);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(result.get("active")).isInstanceOf(MetadataValue.Bool.class);
      assertThat(((MetadataValue.Bool) result.get("active")).value()).isTrue();
    }

    @Test
    @SuppressWarnings("unchecked")
    void listOfStringsValue() {
      Map<String, Object> springMd = Map.of("tags", List.of("a", "b", "c"));
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(result.get("tags")).isInstanceOf(MetadataValue.Tags.class);
      assertThat(((MetadataValue.Tags) result.get("tags")).values()).containsExactly("a", "b", "c");
    }

    @Test
    void nullValueSkipped() {
      Map<String, Object> springMd = new HashMap<>();
      springMd.put("present", "yes");
      springMd.put("absent", null);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(result).containsKey("present");
      assertThat(result).doesNotContainKey("absent");
    }

    @Test
    void emptyMapReturnsEmpty() {
      assertThat(MetadataConverter.toJavaVectors(Map.of())).isEmpty();
    }

    @Test
    void nullMapReturnsEmpty() {
      assertThat(MetadataConverter.toJavaVectors(null)).isEmpty();
    }

    @Test
    void unknownTypeUsesToString() {
      Object custom =
          new Object() {
            @Override
            public String toString() {
              return "custom-value";
            }
          };
      Map<String, Object> springMd = Map.of("obj", custom);
      Map<String, MetadataValue> result = MetadataConverter.toJavaVectors(springMd);

      assertThat(((MetadataValue.Str) result.get("obj")).value()).isEqualTo("custom-value");
    }
  }

  @Nested
  @Tag("unit")
  class ToSpringAi {

    @Test
    void strToString() {
      Map<String, MetadataValue> jvMd = Map.of("key", new MetadataValue.Str("hello"));
      Map<String, Object> result = MetadataConverter.toSpringAi(jvMd);

      assertThat(result.get("key")).isEqualTo("hello");
    }

    @Test
    void integralNumToLong() {
      Map<String, MetadataValue> jvMd = Map.of("count", new MetadataValue.Num(42.0));
      Map<String, Object> result = MetadataConverter.toSpringAi(jvMd);

      assertThat(result.get("count")).isInstanceOf(Long.class);
      assertThat(result.get("count")).isEqualTo(42L);
    }

    @Test
    void fractionalNumToDouble() {
      Map<String, MetadataValue> jvMd = Map.of("pi", new MetadataValue.Num(3.14));
      Map<String, Object> result = MetadataConverter.toSpringAi(jvMd);

      assertThat(result.get("pi")).isInstanceOf(Double.class);
      assertThat(result.get("pi")).isEqualTo(3.14);
    }

    @Test
    void boolToBoolean() {
      Map<String, MetadataValue> jvMd = Map.of("flag", new MetadataValue.Bool(true));
      Map<String, Object> result = MetadataConverter.toSpringAi(jvMd);

      assertThat(result.get("flag")).isInstanceOf(Boolean.class);
      assertThat(result.get("flag")).isEqualTo(true);
    }

    @Test
    void tagsToList() {
      Map<String, MetadataValue> jvMd = Map.of("tags", new MetadataValue.Tags(List.of("x", "y")));
      Map<String, Object> result = MetadataConverter.toSpringAi(jvMd);

      assertThat(result.get("tags")).isInstanceOf(List.class);
      assertThat(result.get("tags")).isEqualTo(List.of("x", "y"));
    }

    @Test
    void emptyMapReturnsEmpty() {
      assertThat(MetadataConverter.toSpringAi(Map.of())).isEmpty();
    }

    @Test
    void nullMapReturnsEmpty() {
      assertThat(MetadataConverter.toSpringAi(null)).isEmpty();
    }
  }

  @Nested
  @Tag("unit")
  class RoundTrip {

    @Test
    void stringRoundTrip() {
      Map<String, Object> original = Map.of("name", "alice");
      Map<String, Object> result =
          MetadataConverter.toSpringAi(MetadataConverter.toJavaVectors(original));

      assertThat(result).isEqualTo(original);
    }

    @Test
    void mixedTypesRoundTrip() {
      Map<String, Object> original = new HashMap<>();
      original.put("str", "hello");
      original.put("num", 42L);
      original.put("dbl", 3.14);
      original.put("flag", true);
      original.put("tags", List.of("a", "b"));

      Map<String, Object> result =
          MetadataConverter.toSpringAi(MetadataConverter.toJavaVectors(original));

      assertThat(result).isEqualTo(original);
    }
  }
}
