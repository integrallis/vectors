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
 * <p>Each {@link Route} has reference utterances whose embeddings are pre-computed and
 * L2-normalized at construction time. Routing a query computes its embedding, L2-normalizes it
 * once, then finds the closest reference across all routes. If the cosine distance is within the
 * route's threshold, the query is classified to that route.
 *
 * <p>Because both the reference embeddings and the query embedding are L2-normalized, cosine
 * similarity reduces to the dot product, so the inner loop uses {@link
 * VectorUtil#dotProduct(float[], float[])} (SIMD-accelerated) and computes distance as {@code 1 -
 * dot}.
 */
public final class SemanticRouter {

  private final EmbeddingFunction embeddingFunction;
  private final Map<String, Route> routes;
  private final Map<String, List<float[]>> routeEmbeddings;
  private final int dimension;

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
    int detectedDimension = -1;

    for (Route route : routes) {
      // Fail fast on a duplicate route name: a second same-named route would otherwise silently
      // overwrite the first here and its embeddings below, discarding exemplars we already paid to
      // embed. A duplicate name is a caller error, not a valid last-wins configuration.
      if (this.routes.containsKey(route.getName())) {
        throw new IllegalArgumentException("duplicate route name: '" + route.getName() + "'");
      }
      this.routes.put(route.getName(), route);
      List<float[]> embeddings = new ArrayList<>();
      for (String reference : route.getReferences()) {
        float[] embedding =
            requireEmbedding("reference '" + reference + "'", embeddingFunction.embed(reference));
        if (detectedDimension < 0) {
          detectedDimension = embedding.length;
        } else {
          requireDimension("reference '" + reference + "'", embedding, detectedDimension);
        }
        // L2-normalize once at construction so the route() inner loop can use a plain dot
        // product (cosine of unit vectors == dot product). throwOnZero=false preserves prior
        // behavior for degenerate zero embeddings (they simply never win the argmin).
        embeddings.add(VectorUtil.l2normalize(embedding, false));
      }
      routeEmbeddings.put(route.getName(), embeddings);
    }
    this.dimension = detectedDimension;
  }

  /**
   * Routes a query to the best matching route.
   *
   * @param query the user query
   * @return RouteMatch with the matched route, or a no-match result
   */
  public RouteMatch route(String query) {
    float[] queryVec = embeddingFunction.embed(query);
    if (dimension < 0) {
      return RouteMatch.noMatch();
    }
    return route(queryVec);
  }

  /**
   * Routes a pre-computed query embedding to the best matching route.
   *
   * <p>The embedding is L2-normalized on a defensive copy, so the caller's array is left unchanged.
   *
   * @param queryEmbedding the query embedding (dimension must match the reference embeddings)
   * @return RouteMatch with the matched route, or a no-match result
   */
  public RouteMatch route(float[] queryEmbedding) {
    if (dimension < 0) {
      return RouteMatch.noMatch();
    }
    requireDimension("query", requireEmbedding("query", queryEmbedding), dimension);

    // Normalize once per route() on a copy (references are already normalized); cosine == dot.
    float[] queryVec = VectorUtil.l2normalize(queryEmbedding.clone(), false);

    String bestRoute = null;
    double bestDistance = Double.MAX_VALUE;

    for (var entry : routeEmbeddings.entrySet()) {
      String routeName = entry.getKey();
      for (float[] refVec : entry.getValue()) {
        double distance = 1.0 - VectorUtil.dotProduct(queryVec, refVec);
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

  private static float[] requireEmbedding(String source, float[] embedding) {
    if (embedding == null) {
      throw new IllegalArgumentException(source + " embedding must not be null");
    }
    if (embedding.length == 0) {
      throw new IllegalArgumentException(source + " embedding must not be empty");
    }
    return embedding;
  }

  private static void requireDimension(String source, float[] embedding, int expectedDimension) {
    if (embedding.length != expectedDimension) {
      throw new IllegalArgumentException(
          source
              + " embedding dimension "
              + embedding.length
              + " != expected "
              + expectedDimension);
    }
  }
}
