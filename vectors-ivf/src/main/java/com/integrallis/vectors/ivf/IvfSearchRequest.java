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
 * Search parameters for {@link IvfIndex#search}.
 *
 * @param query query vector; must have the same dimension as the index
 * @param k number of top results to return
 * @param nprobe number of clusters to probe (higher = better recall, slower)
 * @param gamma SOAR boundary expansion at search time (0.0 = disabled)
 * @param minScore minimum score threshold; hits below this score are excluded
 * @param rescoreFactor candidate-pool multiplier for PQ rescoring. Ignored when the index has no
 *     product quantizer. Must be &ge; 1; a value of 1 disables rescore (top-k ADC scores are
 *     returned as-is). Typical values: 2–8 for high recall. The approximate (ADC) stage collects
 *     {@code k * rescoreFactor} candidates which are then re-ranked against full-precision vectors
 *     to produce the final top-k.
 */
public record IvfSearchRequest(
    float[] query, int k, int nprobe, float gamma, float minScore, int rescoreFactor) {

  /** Validates parameters at construction time. */
  public IvfSearchRequest {
    if (query == null) throw new IllegalArgumentException("query must not be null");
    if (k < 1) throw new IllegalArgumentException("k must be >= 1, got " + k);
    if (nprobe < 1) throw new IllegalArgumentException("nprobe must be >= 1, got " + nprobe);
    if (gamma < 0f) throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
    if (rescoreFactor < 1)
      throw new IllegalArgumentException("rescoreFactor must be >= 1, got " + rescoreFactor);
  }

  /**
   * Backward-compatible 5-arg constructor: rescore disabled ({@code rescoreFactor = 1}). Existing
   * IVF-flat callers continue to compile unchanged.
   */
  public IvfSearchRequest(float[] query, int k, int nprobe, float gamma, float minScore) {
    this(query, k, nprobe, gamma, minScore, 1);
  }

  /** Convenience factory with SOAR disabled, no score threshold, and rescore disabled. */
  public static IvfSearchRequest of(float[] query, int k, int nprobe) {
    return new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE, 1);
  }

  /**
   * Convenience factory with SOAR disabled, no score threshold, and the given rescore factor. Use
   * this when searching a PQ-enabled {@link IvfIndex} to trade latency for recall.
   */
  public static IvfSearchRequest of(float[] query, int k, int nprobe, int rescoreFactor) {
    return new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE, rescoreFactor);
  }
}
