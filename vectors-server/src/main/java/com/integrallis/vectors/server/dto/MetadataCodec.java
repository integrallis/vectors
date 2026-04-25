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
package com.integrallis.vectors.server.dto;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.MetadataValue.Bool;
import com.integrallis.vectors.db.MetadataValue.Num;
import com.integrallis.vectors.db.MetadataValue.Str;
import com.integrallis.vectors.db.MetadataValue.Tags;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Bidirectional mapping between JSON and the typed {@link MetadataValue} lattice used by the
 * embedded collection.
 *
 * <p>Wire mapping (JSON → MetadataValue):
 *
 * <ul>
 *   <li>string → {@link Str}
 *   <li>number → {@link Num} (double)
 *   <li>boolean → {@link Bool}
 *   <li>array of strings → {@link Tags}
 *   <li>null → skipped (the key is dropped from the resulting map)
 * </ul>
 *
 * <p>Any other JSON type (object, mixed-type array, array of non-strings) produces an {@link
 * IllegalArgumentException} describing the offending field.
 */
public final class MetadataCodec {

  private MetadataCodec() {}

  /**
   * Decodes a JSON object into a typed metadata map.
   *
   * @param node the JSON object, or {@code null}/missing for an empty map
   * @throws IllegalArgumentException if any value has an unsupported JSON type
   */
  public static Map<String, MetadataValue> fromJson(JsonNode node) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return Map.of();
    }
    if (!node.isObject()) {
      throw new IllegalArgumentException(
          "metadata must be a JSON object, got " + node.getNodeType());
    }
    Map<String, MetadataValue> out = new HashMap<>();
    for (Map.Entry<String, JsonNode> e : node.properties()) {
      MetadataValue mv = decodeValue(e.getKey(), e.getValue());
      if (mv != null) {
        out.put(e.getKey(), mv);
      }
    }
    return Map.copyOf(out);
  }

  private static MetadataValue decodeValue(String key, JsonNode v) {
    if (v == null || v.isNull()) {
      return null;
    }
    if (v.isTextual()) {
      return new Str(v.asText());
    }
    if (v.isBoolean()) {
      return new Bool(v.asBoolean());
    }
    if (v.isNumber()) {
      return Num.of(v.asDouble());
    }
    if (v.isArray()) {
      List<String> tags = new ArrayList<>(v.size());
      for (JsonNode elem : v) {
        if (!elem.isTextual()) {
          throw new IllegalArgumentException(
              "metadata tags array for '" + key + "' must contain only strings");
        }
        tags.add(elem.asText());
      }
      return new Tags(tags);
    }
    throw new IllegalArgumentException(
        "metadata value for '" + key + "' has unsupported type: " + v.getNodeType());
  }

  /** Encodes a typed metadata map into a plain JSON object node (no type envelope). */
  public static ObjectNode toJson(Map<String, MetadataValue> metadata) {
    ObjectNode out = JsonNodeFactory.instance.objectNode();
    if (metadata == null) {
      return out;
    }
    for (Map.Entry<String, MetadataValue> e : metadata.entrySet()) {
      out.set(e.getKey(), encodeValue(e.getValue()));
    }
    return out;
  }

  private static JsonNode encodeValue(MetadataValue v) {
    JsonNodeFactory f = JsonNodeFactory.instance;
    return switch (v) {
      case Str s -> f.textNode(s.value());
      case Num n -> f.numberNode(n.value());
      case Bool b -> f.booleanNode(b.value());
      case Tags t -> {
        ArrayNode arr = f.arrayNode(t.values().size());
        for (String tag : t.values()) {
          arr.add(tag);
        }
        yield arr;
      }
    };
  }
}
