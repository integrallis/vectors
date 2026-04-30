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
package com.integrallis.vectors.hybrid.text;

import java.util.Objects;

/**
 * Result of a full-text search: parallel arrays of document IDs and their relevance scores.
 *
 * @param ids document identifiers, ordered by descending score
 * @param scores corresponding relevance scores
 */
public record TextSearchOutcome(String[] ids, float[] scores) {

  public TextSearchOutcome {
    Objects.requireNonNull(ids, "ids");
    Objects.requireNonNull(scores, "scores");
    if (ids.length != scores.length) {
      throw new IllegalArgumentException(
          "ids.length (" + ids.length + ") != scores.length (" + scores.length + ")");
    }
  }

  /** Returns the number of results. */
  public int size() {
    return ids.length;
  }

  /** Returns an empty outcome. */
  public static TextSearchOutcome empty() {
    return new TextSearchOutcome(new String[0], new float[0]);
  }
}
