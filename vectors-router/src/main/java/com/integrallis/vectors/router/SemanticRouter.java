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

import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Local in-memory semantic router using cosine similarity.
 *
 * <p>Each {@link Route} has reference utterances whose embeddings are pre-computed at construction
 * time. Routing a query computes its embedding, then finds the closest reference across all routes.
 * If the cosine distance is within the route's threshold, the query is classified to that route.
 *
 * <p>Distance is computed via {@link VectorUtil#cosine(float[], float[])} (SIMD-accelerated) as
 * {@code 1 - similarity}.
 */
public final class SemanticRouter {

  private final EmbeddingFunction embeddingFunction;
  private final Map<String, Route> routes;
  private final Map<String, List<float[]>> routeEmbeddings;

  /**
   * Creates a semantic router.
   *
   * @param embeddingFunction function for computing embeddings
   * @param routes list of routes to register
   */
  public SemanticRouter(EmbeddingFunction embeddingFunction, List<Route> routes) {
    this.embeddingFunction = Objects.requireNonNull(embeddingFunction, "embeddingFunction");
    Objects.requireNonNull(routes, "routes");
    this.routes = new LinkedHashMap<>();
    this.routeEmbeddings = new LinkedHashMap<>();

    for (Route route : routes) {
      this.routes.put(route.getName(), route);
      List<float[]> embeddings = new ArrayList<>();
      for (String reference : route.getReferences()) {
        embeddings.add(embeddingFunction.embed(reference));
      }
      routeEmbeddings.put(route.getName(), embeddings);
    }
  }

  /**
   * Routes a query to the best matching route.
   *
   * @param query the user query
   * @return RouteMatch with the matched route, or a no-match result
   */
  public RouteMatch route(String query) {
    float[] queryVec = embeddingFunction.embed(query);

    String bestRoute = null;
    double bestDistance = Double.MAX_VALUE;

    for (var entry : routeEmbeddings.entrySet()) {
      String routeName = entry.getKey();
      for (float[] refVec : entry.getValue()) {
        double distance = 1.0 - VectorUtil.cosine(queryVec, refVec);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestRoute = routeName;
        }
      }
    }

    // Check if the best match is within its route's threshold
    if (bestRoute != null) {
      Route matched = routes.get(bestRoute);
      if (bestDistance <= matched.getDistanceThreshold()) {
        return RouteMatch.of(bestRoute, bestDistance);
      }
    }

    return RouteMatch.noMatch();
  }

  /** Returns the names of all registered routes. */
  public List<String> getRouteNames() {
    return new ArrayList<>(routes.keySet());
  }

  /** Gets a route by name, or {@code null} if not found. */
  public Route get(String name) {
    return routes.get(name);
  }
}
