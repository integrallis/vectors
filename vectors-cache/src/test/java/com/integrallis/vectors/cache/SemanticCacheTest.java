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
package com.integrallis.vectors.cache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SemanticCacheTest {

  @Test
  void entryDefensivelyCopiesEmbeddingOnConstruction() {
    float[] embedding = {1f, 2f, 3f};
    SemanticCache.Entry<String> entry = new SemanticCache.Entry<>("k", embedding, "v");

    embedding[0] = 9f;

    assertThat(entry.embedding()).containsExactly(1f, 2f, 3f);
  }

  @Test
  void entryDefensivelyCopiesEmbeddingOnAccess() {
    SemanticCache.Entry<String> entry =
        new SemanticCache.Entry<>("k", new float[] {1f, 2f, 3f}, "v");

    float[] first = entry.embedding();
    first[0] = 9f;

    assertThat(entry.embedding()).containsExactly(1f, 2f, 3f);
  }
}
