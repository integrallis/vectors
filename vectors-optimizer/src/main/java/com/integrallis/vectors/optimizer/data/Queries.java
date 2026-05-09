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
package com.integrallis.vectors.optimizer.data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Set of evaluation queries: {@code queryId -> queryText}. Two on-disk shapes are accepted:
 *
 * <ul>
 *   <li>Object form: {@code {"q1": "what is X", "q2": "..."}} preserves caller-supplied IDs.
 *   <li>Array form: {@code ["query a", "query b", ...]} auto-numbers {@code q0}, {@code q1}, ...
 * </ul>
 */
public record Queries(Map<String, String> byId) {

  public Queries {
    Objects.requireNonNull(byId, "byId");
    Map<String, String> copy = new LinkedHashMap<>();
    for (var e : byId.entrySet()) {
      Objects.requireNonNull(e.getKey(), "queryId");
      Objects.requireNonNull(e.getValue(), "queryText for " + e.getKey());
      copy.put(e.getKey(), e.getValue());
    }
    byId = Map.copyOf(copy);
  }

  /** Number of queries. */
  public int size() {
    return byId.size();
  }

  /** Loads queries from JSON in either object form or array form. */
  public static Queries loadJson(Path path) {
    Objects.requireNonNull(path, "path");
    ObjectMapper mapper = new ObjectMapper();
    try {
      JsonNode root = mapper.readTree(Files.readAllBytes(path));
      Map<String, String> out = new LinkedHashMap<>();
      if (root.isObject()) {
        var it = root.fields();
        while (it.hasNext()) {
          var f = it.next();
          out.put(f.getKey(), f.getValue().asText());
        }
      } else if (root.isArray()) {
        for (int i = 0; i < root.size(); i++) {
          out.put("q" + i, root.get(i).asText());
        }
      } else {
        throw new IllegalArgumentException(
            "Queries JSON must be an object or array; got " + root.getNodeType() + " at " + path);
      }
      return new Queries(out);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to read queries from " + path, ioe);
    }
  }
}
