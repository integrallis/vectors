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
package com.integrallis.vectors.optimizer.embed;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Strategy for converting query text to dense vectors. The optimizer module ships no real
 * implementation: callers (typically the Studio web tier) inject one that delegates to the same
 * embedder used to populate the collection under test.
 */
@FunctionalInterface
public interface EmbeddingProvider {

  /** Embeds a single query string. */
  float[] embed(String text);

  /** Default batched implementation that calls {@link #embed} in a loop. */
  default List<float[]> embedAll(List<String> texts) {
    Objects.requireNonNull(texts, "texts");
    List<float[]> out = new ArrayList<>(texts.size());
    for (String t : texts) out.add(embed(t));
    return out;
  }
}
