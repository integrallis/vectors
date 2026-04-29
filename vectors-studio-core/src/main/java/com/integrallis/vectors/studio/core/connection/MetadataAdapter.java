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
package com.integrallis.vectors.studio.core.connection;

import com.fasterxml.jackson.databind.JsonNode;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.MetadataValue.Bool;
import com.integrallis.vectors.core.MetadataValue.Num;
import com.integrallis.vectors.core.MetadataValue.Str;
import com.integrallis.vectors.core.MetadataValue.Tags;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Lossy projections between {@link MetadataValue} maps and plain {@code Map<String, Object>}. */
final class MetadataAdapter {

  private MetadataAdapter() {}

  /** Projects a typed metadata map into untyped form for transport to the UI. */
  static Map<String, Object> toMap(Map<String, MetadataValue> in) {
    if (in == null || in.isEmpty()) return Map.of();
    Map<String, Object> out = new HashMap<>(in.size());
    for (Map.Entry<String, MetadataValue> e : in.entrySet()) {
      out.put(e.getKey(), toObject(e.getValue()));
    }
    return out;
  }

  private static Object toObject(MetadataValue v) {
    return switch (v) {
      case Str s -> s.value();
      case Num n -> n.value();
      case Bool b -> b.value();
      case Tags t -> t.values();
    };
  }

  /** Projects a {@link JsonNode} into a plain object map for the UI layer. */
  static Map<String, Object> fromJsonNode(JsonNode node) {
    if (node == null || !node.isObject()) return Map.of();
    Map<String, Object> out = new HashMap<>();
    for (Map.Entry<String, JsonNode> e : node.properties()) {
      out.put(e.getKey(), jsonValue(e.getValue()));
    }
    return out;
  }

  private static Object jsonValue(JsonNode v) {
    if (v == null || v.isNull()) return null;
    if (v.isTextual()) return v.asText();
    if (v.isBoolean()) return v.asBoolean();
    if (v.isNumber()) return v.asDouble();
    if (v.isArray()) {
      List<Object> arr = new ArrayList<>(v.size());
      for (JsonNode el : v) arr.add(jsonValue(el));
      return arr;
    }
    return v.toString();
  }
}
