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
package com.integrallis.vectors.ivf;

/**
 * A single result from an {@link IvfIndex} search.
 *
 * @param ordinal row index in the original {@code float[][] vectors} array passed to {@link
 *     IvfIndex#build}
 * @param id document identifier supplied at build time; {@code null} if no ids were provided
 * @param score similarity score; higher is more similar (dot product or negated L2)
 */
public record IvfHit(int ordinal, String id, float score) implements Comparable<IvfHit> {

  @Override
  public int compareTo(IvfHit other) {
    // Descending score: higher score = better hit = earlier in list
    return Float.compare(other.score, this.score);
  }
}
