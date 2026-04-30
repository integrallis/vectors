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

/**
 * Result of routing a query through a {@link SemanticRouter}.
 *
 * <p>If no route matched, {@link #getName()} returns {@code null}.
 */
public final class RouteMatch {

  private final String name;
  private final Double distance;

  private RouteMatch(String name, Double distance) {
    this.name = name;
    this.distance = distance;
  }

  /** Returns the matched route name, or {@code null} if no route matched. */
  public String getName() {
    return name;
  }

  /** Returns the distance to the best matching reference, or {@code null} if no match. */
  public Double getDistance() {
    return distance;
  }

  /** Creates a "no match" result. */
  public static RouteMatch noMatch() {
    return new RouteMatch(null, null);
  }

  /** Creates a match result. */
  public static RouteMatch of(String name, double distance) {
    return new RouteMatch(name, distance);
  }
}
