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

import com.integrallis.vectors.db.MetadataValue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Converts between Spring AI metadata ({@code Map<String, Object>}) and java-vectors typed metadata
 * ({@code Map<String, MetadataValue>}).
 *
 * <p>This class is stateless; all methods are static.
 */
final class MetadataConverter {

  private MetadataConverter() {}

  /**
   * Converts Spring AI metadata to java-vectors metadata.
   *
   * <p>Type mapping:
   *
   * <ul>
   *   <li>{@code String} &rarr; {@link MetadataValue.Str}
   *   <li>{@code Number} &rarr; {@link MetadataValue.Num} (widened to double)
   *   <li>{@code Boolean} &rarr; {@link MetadataValue.Bool}
   *   <li>{@code List<String>} &rarr; {@link MetadataValue.Tags}
   *   <li>{@code null} &rarr; skipped
   *   <li>Other &rarr; {@link MetadataValue.Str} via {@code toString()}
   * </ul>
   *
   * @param springMetadata Spring AI metadata map, may be null
   * @return java-vectors metadata map, never null
   */
  @SuppressWarnings("unchecked")
  static Map<String, MetadataValue> toJavaVectors(Map<String, Object> springMetadata) {
    if (springMetadata == null || springMetadata.isEmpty()) {
      return Map.of();
    }
    Map<String, MetadataValue> result = new LinkedHashMap<>(springMetadata.size());
    for (Map.Entry<String, Object> entry : springMetadata.entrySet()) {
      Object value = entry.getValue();
      if (value == null) {
        continue;
      }
      MetadataValue mv =
          switch (value) {
            case String s -> new MetadataValue.Str(s);
            case Boolean b -> new MetadataValue.Bool(b);
            case Integer i -> new MetadataValue.Num(i.doubleValue());
            case Long l -> new MetadataValue.Num(l.doubleValue());
            case Float f -> new MetadataValue.Num(f.doubleValue());
            case Double d -> new MetadataValue.Num(d);
            case List<?> list ->
                new MetadataValue.Tags(list.stream().map(Object::toString).toList());
            default -> new MetadataValue.Str(value.toString());
          };
      result.put(entry.getKey(), mv);
    }
    return Map.copyOf(result);
  }

  /**
   * Converts java-vectors metadata to Spring AI metadata.
   *
   * <p>Type mapping:
   *
   * <ul>
   *   <li>{@link MetadataValue.Str} &rarr; {@code String}
   *   <li>{@link MetadataValue.Num} &rarr; {@code Long} (if integral) or {@code Double}
   *   <li>{@link MetadataValue.Bool} &rarr; {@code Boolean}
   *   <li>{@link MetadataValue.Tags} &rarr; {@code List<String>}
   * </ul>
   *
   * @param jvMetadata java-vectors metadata map, may be null
   * @return Spring AI metadata map, never null
   */
  static Map<String, Object> toSpringAi(Map<String, MetadataValue> jvMetadata) {
    if (jvMetadata == null || jvMetadata.isEmpty()) {
      return Map.of();
    }
    Map<String, Object> result = new LinkedHashMap<>(jvMetadata.size());
    for (Map.Entry<String, MetadataValue> entry : jvMetadata.entrySet()) {
      Object value =
          switch (entry.getValue()) {
            case MetadataValue.Str s -> s.value();
            case MetadataValue.Num n -> {
              double d = n.value();
              yield (d == Math.floor(d) && !Double.isInfinite(d))
                  ? (Object) Long.valueOf((long) d)
                  : Double.valueOf(d);
            }
            case MetadataValue.Bool b -> b.value();
            case MetadataValue.Tags t -> t.values();
          };
      result.put(entry.getKey(), value);
    }
    return Map.copyOf(result);
  }
}
