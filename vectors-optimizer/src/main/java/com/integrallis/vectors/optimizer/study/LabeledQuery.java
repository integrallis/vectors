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
package com.integrallis.vectors.optimizer.study;

import java.util.Objects;

/**
 * A single labelled probe used by {@link RouterThresholdStudy} and {@link CacheThresholdStudy}.
 *
 * <p>{@code expectedLabel == null} means "the system should produce no match" — for the router, the
 * route falls through; for the cache, the lookup misses.
 *
 * @param text the query string (router uses its own embedder)
 * @param embedding the pre-computed embedding (may be {@code null} for router-only flows)
 * @param expectedLabel the expected route name / cache key, or {@code null} for an expected miss
 */
public record LabeledQuery(String text, float[] embedding, String expectedLabel) {

  public LabeledQuery {
    Objects.requireNonNull(text, "text");
  }

  /** Convenience constructor for router studies that re-embed the text on every trial. */
  public static LabeledQuery routerProbe(String text, String expectedRoute) {
    return new LabeledQuery(text, null, expectedRoute);
  }

  /** Convenience constructor for cache studies that pass a pre-computed embedding. */
  public static LabeledQuery cacheProbe(String text, float[] embedding, String expectedKey) {
    return new LabeledQuery(text, embedding, expectedKey);
  }
}
