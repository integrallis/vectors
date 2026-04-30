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
package com.integrallis.vectors.router;

import java.util.List;
import java.util.Objects;

/**
 * A semantic route with reference utterances and a distance threshold.
 *
 * <p>Routes are used by {@link SemanticRouter} to classify queries into categories (e.g. off-topic,
 * PII requests) by computing cosine similarity against the reference embeddings.
 */
public final class Route {

  private final String name;
  private final List<String> references;
  private final double distanceThreshold;

  private Route(String name, List<String> references, double distanceThreshold) {
    this.name = Objects.requireNonNull(name, "name");
    this.references = List.copyOf(Objects.requireNonNull(references, "references"));
    this.distanceThreshold = distanceThreshold;
  }

  public String getName() {
    return name;
  }

  public List<String> getReferences() {
    return references;
  }

  public double getDistanceThreshold() {
    return distanceThreshold;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private String name;
    private List<String> references = List.of();
    private double distanceThreshold = 0.3;

    public Builder name(String name) {
      this.name = name;
      return this;
    }

    public Builder references(List<String> references) {
      this.references = references;
      return this;
    }

    public Builder distanceThreshold(double distanceThreshold) {
      this.distanceThreshold = distanceThreshold;
      return this;
    }

    public Route build() {
      return new Route(name, references, distanceThreshold);
    }
  }
}
