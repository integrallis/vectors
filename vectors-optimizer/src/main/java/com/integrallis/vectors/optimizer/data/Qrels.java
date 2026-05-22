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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Query-relevance judgements: {@code queryId -> docId -> integer relevance grade}.
 *
 * <p>Layout matches TREC-style qrels and the {@code redis-retrieval-optimizer} schema. A grade of 0
 * (or absence from the inner map) means "not relevant"; positive grades indicate relevance with
 * higher numbers ranking better (used by NDCG).
 */
public record Qrels(Map<String, Map<String, Integer>> relevance) {

  public Qrels {
    Objects.requireNonNull(relevance, "relevance");
    Map<String, Map<String, Integer>> copy = new LinkedHashMap<>();
    for (var e : relevance.entrySet()) {
      Objects.requireNonNull(e.getKey(), "queryId");
      Objects.requireNonNull(e.getValue(), "docs");
      Map<String, Integer> inner = new LinkedHashMap<>();
      for (var d : e.getValue().entrySet()) {
        Objects.requireNonNull(d.getKey(), "docId");
        Integer grade = d.getValue();
        Objects.requireNonNull(grade, "grade");
        if (grade < 0) {
          throw new IllegalArgumentException(
              "Negative relevance grade for query "
                  + e.getKey()
                  + " doc "
                  + d.getKey()
                  + ": "
                  + grade);
        }
        inner.put(d.getKey(), grade);
      }
      copy.put(e.getKey(), Map.copyOf(inner));
    }
    relevance = Map.copyOf(copy);
  }

  /** Loads qrels from a JSON file shaped {@code {"q1": {"d1": 1, "d2": 0}, ...}}. */
  public static Qrels loadJson(Path path) {
    Objects.requireNonNull(path, "path");
    ObjectMapper mapper = new ObjectMapper();
    try {
      Map<String, Map<String, Integer>> raw =
          mapper.readValue(
              Files.readAllBytes(path), new TypeReference<Map<String, Map<String, Integer>>>() {});
      return new Qrels(raw);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to read qrels from " + path, ioe);
    }
  }

  /** Total number of (query, doc) judgements. */
  public int size() {
    int n = 0;
    for (Map<String, Integer> inner : relevance.values()) {
      n += inner.size();
    }
    return n;
  }
}
