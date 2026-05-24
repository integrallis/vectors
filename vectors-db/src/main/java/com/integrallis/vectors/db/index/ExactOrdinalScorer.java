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
package com.integrallis.vectors.db.index;

/** Creates exact full-precision scorers for ordinal-based rescoring. */
interface ExactOrdinalScorer {

  /**
   * Creates a scorer for one query. The returned scorer may keep per-query scratch state and is not
   * required to be thread-safe.
   *
   * @param query full-precision query vector
   * @return scorer for stored ordinals
   */
  OrdinalScorer exactScorerFor(float[] query);

  @FunctionalInterface
  interface OrdinalScorer {
    float score(int ordinal);
  }
}
