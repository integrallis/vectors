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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SemanticRouterTest {

  // Fixed embeddings for deterministic tests.
  // "sports" cluster around [1,0,0], "food" cluster around [0,1,0], unrelated around [0,0,1].
  private static final Map<String, float[]> EMBEDDINGS =
      Map.ofEntries(
          Map.entry("football game", new float[] {0.95f, 0.05f, 0.0f}),
          Map.entry("basketball score", new float[] {0.90f, 0.10f, 0.0f}),
          Map.entry("soccer match", new float[] {0.92f, 0.08f, 0.0f}),
          Map.entry("pasta recipe", new float[] {0.05f, 0.95f, 0.0f}),
          Map.entry("sushi restaurant", new float[] {0.08f, 0.90f, 0.02f}),
          Map.entry("baking bread", new float[] {0.03f, 0.93f, 0.04f}),
          // Queries
          Map.entry("who won the game?", new float[] {0.93f, 0.07f, 0.0f}),
          Map.entry("best pizza nearby", new float[] {0.06f, 0.92f, 0.02f}),
          Map.entry("meaning of life", new float[] {0.1f, 0.1f, 0.80f}),
          Map.entry("near threshold", new float[] {0.5f, 0.5f, 0.0f}));

  private static final EmbeddingFunction FAKE_EMBEDDER = text -> EMBEDDINGS.get(text);

  private static final Route SPORTS =
      Route.builder()
          .name("sports")
          .references(List.of("football game", "basketball score", "soccer match"))
          .distanceThreshold(0.2)
          .build();

  private static final Route FOOD =
      Route.builder()
          .name("food")
          .references(List.of("pasta recipe", "sushi restaurant", "baking bread"))
          .distanceThreshold(0.2)
          .build();

  @Nested
  class Routing {

    @Test
    void matchesSportsRoute() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      RouteMatch match = router.route("who won the game?");

      assertThat(match.getName()).isEqualTo("sports");
      assertThat(match.getDistance()).isNotNull().isLessThan(0.2);
    }

    @Test
    void matchesFoodRoute() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      RouteMatch match = router.route("best pizza nearby");

      assertThat(match.getName()).isEqualTo("food");
      assertThat(match.getDistance()).isNotNull().isLessThan(0.2);
    }

    @Test
    void noMatchWhenDistantFromAllRoutes() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      RouteMatch match = router.route("meaning of life");

      assertThat(match.getName()).isNull();
      assertThat(match.getDistance()).isNull();
    }

    @Test
    void noMatchWhenAboveThreshold() {
      // "near threshold" is equidistant — cosine distance > 0.2 to both clusters
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      RouteMatch match = router.route("near threshold");

      assertThat(match.getName()).isNull();
    }
  }

  @Nested
  class Metadata {

    @Test
    void getRouteNamesReturnsAllNames() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      assertThat(router.getRouteNames()).containsExactly("sports", "food");
    }

    @Test
    void getByNameReturnsRoute() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS, FOOD));

      assertThat(router.get("sports")).isSameAs(SPORTS);
      assertThat(router.get("food")).isSameAs(FOOD);
    }

    @Test
    void getByNameReturnsNullForUnknown() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS));

      assertThat(router.get("nonexistent")).isNull();
    }
  }

  @Nested
  class Validation {

    @Test
    void nullEmbeddingFunctionThrows() {
      assertThatThrownBy(() -> new SemanticRouter(null, List.of(SPORTS)))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("embeddingFunction");
    }

    @Test
    void nullRoutesThrows() {
      assertThatThrownBy(() -> new SemanticRouter(FAKE_EMBEDDER, null))
          .isInstanceOf(NullPointerException.class);
    }

    @Test
    void nullReferenceEmbeddingThrowsAtConstruction() {
      assertThatThrownBy(() -> new SemanticRouter(text -> null, List.of(SPORTS)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("reference 'football game' embedding must not be null");
    }

    @Test
    void emptyReferenceEmbeddingThrowsAtConstruction() {
      assertThatThrownBy(() -> new SemanticRouter(text -> new float[0], List.of(SPORTS)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("reference 'football game' embedding must not be empty");
    }

    @Test
    void wrongSizedReferenceEmbeddingThrowsAtConstruction() {
      EmbeddingFunction embedder =
          text -> text.equals("football game") ? new float[] {1, 0, 0} : new float[] {1, 0};

      assertThatThrownBy(() -> new SemanticRouter(embedder, List.of(SPORTS)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("reference 'basketball score' embedding dimension 2 != expected 3");
    }

    @Test
    void nullQueryEmbeddingThrowsBeforeSimilarity() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(SPORTS));

      assertThatThrownBy(() -> router.route("missing query"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("query embedding must not be null");
    }

    @Test
    void wrongSizedQueryEmbeddingThrowsBeforeSimilarity() {
      EmbeddingFunction embedder =
          text -> text.equals("wrong sized query") ? new float[] {1, 0} : EMBEDDINGS.get(text);
      SemanticRouter router = new SemanticRouter(embedder, List.of(SPORTS));

      assertThatThrownBy(() -> router.route("wrong sized query"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("query embedding dimension 2 != expected 3");
    }
  }

  @Nested
  class EdgeCases {

    @Test
    void emptyRoutesListReturnsNoMatch() {
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of());

      RouteMatch match = router.route("who won the game?");

      assertThat(match.getName()).isNull();
    }

    @Test
    void singleRouteWithSingleReference() {
      Route single =
          Route.builder()
              .name("sports")
              .references(List.of("football game"))
              .distanceThreshold(0.2)
              .build();
      SemanticRouter router = new SemanticRouter(FAKE_EMBEDDER, List.of(single));

      RouteMatch match = router.route("who won the game?");

      assertThat(match.getName()).isEqualTo("sports");
    }
  }
}
