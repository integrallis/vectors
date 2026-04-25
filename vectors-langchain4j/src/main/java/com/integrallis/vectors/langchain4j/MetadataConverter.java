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

import com.integrallis.vectors.db.MetadataValue;
import dev.langchain4j.data.document.Metadata;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Converts between LangChain4j {@link Metadata} and java-vectors typed metadata ({@code Map<String,
 * MetadataValue>}).
 *
 * <p>This class is stateless; all methods are static.
 *
 * <p>Type mapping (LangChain4j &rarr; java-vectors):
 *
 * <ul>
 *   <li>{@code String} &rarr; {@link MetadataValue.Str}
 *   <li>{@code Integer}/{@code Long} &rarr; {@link MetadataValue.Num} (widened to double)
 *   <li>{@code Float}/{@code Double} &rarr; {@link MetadataValue.Num}
 *   <li>{@code UUID} &rarr; {@link MetadataValue.Str} (via {@code toString()})
 *   <li>{@code null} values &rarr; skipped
 * </ul>
 *
 * <p>Type mapping (java-vectors &rarr; LangChain4j):
 *
 * <ul>
 *   <li>{@link MetadataValue.Str} &rarr; {@code String}
 *   <li>{@link MetadataValue.Num} &rarr; {@code Long} (if integral) or {@code Double}
 *   <li>{@link MetadataValue.Bool} &rarr; {@code String} ("true"/"false") since Metadata has no
 *       boolean support
 *   <li>{@link MetadataValue.Tags} &rarr; {@code String} (comma-separated; values containing commas
 *       cannot round-trip since LangChain4j {@link Metadata} has no native list support)
 * </ul>
 */
final class MetadataConverter {

  private MetadataConverter() {}

  /**
   * Converts LangChain4j metadata to java-vectors metadata.
   *
   * @param metadata LangChain4j metadata, may be null
   * @return java-vectors metadata map, never null
   */
  static Map<String, MetadataValue> toJavaVectors(Metadata metadata) {
    if (metadata == null) {
      return Map.of();
    }
    Map<String, Object> map = metadata.toMap();
    if (map.isEmpty()) {
      return Map.of();
    }
    Map<String, MetadataValue> result = new LinkedHashMap<>(map.size());
    for (Map.Entry<String, Object> entry : map.entrySet()) {
      Object value = entry.getValue();
      if (value == null) {
        continue;
      }
      MetadataValue mv =
          switch (value) {
            case String s -> new MetadataValue.Str(s);
            case Integer i -> new MetadataValue.Num(i.doubleValue());
            case Long l -> new MetadataValue.Num(l.doubleValue());
            case Float f -> new MetadataValue.Num(f.doubleValue());
            case Double d -> new MetadataValue.Num(d);
            default -> new MetadataValue.Str(value.toString());
          };
      result.put(entry.getKey(), mv);
    }
    return Map.copyOf(result);
  }

  /**
   * Converts java-vectors metadata to LangChain4j metadata.
   *
   * @param jvMetadata java-vectors metadata map, may be null
   * @return LangChain4j metadata, never null
   */
  static Metadata toLangChain4j(Map<String, MetadataValue> jvMetadata) {
    if (jvMetadata == null || jvMetadata.isEmpty()) {
      return new Metadata();
    }
    Metadata metadata = new Metadata();
    for (Map.Entry<String, MetadataValue> entry : jvMetadata.entrySet()) {
      String key = entry.getKey();
      switch (entry.getValue()) {
        case MetadataValue.Str s -> metadata.put(key, s.value());
        case MetadataValue.Num n -> {
          double d = n.value();
          if (d == Math.floor(d) && !Double.isInfinite(d)) {
            metadata.put(key, (long) d);
          } else {
            metadata.put(key, d);
          }
        }
        case MetadataValue.Bool b -> metadata.put(key, Boolean.toString(b.value()));
        case MetadataValue.Tags t ->
            metadata.put(key, t.values().stream().collect(Collectors.joining(",")));
      }
    }
    return metadata;
  }
}
