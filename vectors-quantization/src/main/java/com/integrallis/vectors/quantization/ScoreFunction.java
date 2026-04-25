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
package com.integrallis.vectors.quantization;

/**
 * Computes the approximate similarity score between a query and a stored vector identified by
 * ordinal. Created by {@link CompressedVectors#scoreFunctionFor}.
 *
 * <p>Implementations are not thread-safe; each thread should create its own instance via {@link
 * CompressedVectors#scoreFunctionFor}.
 */
@FunctionalInterface
public interface ScoreFunction {

  /**
   * Returns the approximate similarity score for the vector at the given ordinal.
   *
   * @param ordinal the 0-based index of the stored vector
   * @return the approximate similarity score (higher means more similar)
   */
  float score(int ordinal);
}
