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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A retrieval run: per-query ranked lists of {@code (docId, score)} pairs. Mirrors {@code
 * redis-retrieval-optimizer.schema.SearchMethodOutput}.
 *
 * <p>Inner maps preserve insertion order (highest-scoring doc first by convention) so that {@code
 * Metrics.recallAtK} et al. can iterate them as ranked lists.
 */
public record Run(Map<String, LinkedHashMap<String, Double>> ranking) {

  public Run {
    Objects.requireNonNull(ranking, "ranking");
    Map<String, LinkedHashMap<String, Double>> copy = new LinkedHashMap<>();
    for (var e : ranking.entrySet()) {
      Objects.requireNonNull(e.getKey(), "queryId");
      Objects.requireNonNull(e.getValue(), "ranking row");
      copy.put(e.getKey(), new LinkedHashMap<>(e.getValue()));
    }
    ranking = Map.copyOf(copy);
  }

  /** Returns a new {@link Builder}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Mutable builder for streaming construction. */
  public static final class Builder {
    private final Map<String, LinkedHashMap<String, Double>> rows = new LinkedHashMap<>();

    /** Append a {@code (docId, score)} entry to the ranked list of {@code queryId}. */
    public Builder add(String queryId, String docId, double score) {
      Objects.requireNonNull(queryId, "queryId");
      Objects.requireNonNull(docId, "docId");
      rows.computeIfAbsent(queryId, k -> new LinkedHashMap<>()).put(docId, score);
      return this;
    }

    public Run build() {
      return new Run(rows);
    }
  }
}
