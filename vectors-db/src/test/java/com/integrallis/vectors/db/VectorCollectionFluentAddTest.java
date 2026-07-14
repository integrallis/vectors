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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The fluent {@code add(id, vector[, text])} overloads on {@link VectorCollection} return the
 * collection so inserts chain into a single expression terminated by {@code commit()}.
 */
@Tag("unit")
class VectorCollectionFluentAddTest {

  @Test
  void fluentAddChainStagesEveryDocumentThenCommits() {
    try (VectorCollection collection =
        VectorCollection.builder()
            .dimension(3)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.HNSW)
            .build()) {

      collection
          .add("a", new float[] {1, 0, 0}, "hello world")
          .add("b", new float[] {0, 1, 0}, "goodbye world")
          .add("c", new float[] {0, 0, 1})
          .commit();

      assertThat(collection.size()).isEqualTo(3);
      assertThat(collection.contains("a")).isTrue();
      assertThat(collection.get("a").text()).isEqualTo("hello world");
      assertThat(collection.get("c").text()).as("id+vector overload leaves text null").isNull();
    }
  }

  @Test
  void fluentAddReturnsTheSameCollectionInstance() {
    try (VectorCollection collection =
        VectorCollection.builder().dimension(2).metric(SimilarityFunction.COSINE).build()) {
      VectorCollection returned = collection.add("x", new float[] {1, 0});
      assertThat(returned).isSameAs(collection);
    }
  }
}
