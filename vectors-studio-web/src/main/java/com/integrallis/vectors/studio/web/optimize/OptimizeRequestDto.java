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
package com.integrallis.vectors.studio.web.optimize;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Wire-format JSON payload accepted by {@code POST /api/optimize/studies}. Only Index studies are
 * exposed in v1; router/cache studies will land in a follow-up. The search-space knobs are
 * deliberately limited (HNSW {@code m} and {@code efConstruction}) so the form fits on one page —
 * power users can extend the space programmatically via the {@code vectors-optimizer} module.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record OptimizeRequestDto(
    String collection,
    String sampler,
    Integer nTrials,
    Integer kForMetrics,
    Long seed,
    Integer querySampleSize,
    String metadataField,
    Integer mMin,
    Integer mMax,
    Integer efMin,
    Integer efMax,
    Double recallWeight,
    Double ndcgWeight,
    Double latencyWeight) {

  /** Defaults applied when a field is null on the wire. */
  public OptimizeRequestDto withDefaults() {
    return new OptimizeRequestDto(
        collection,
        sampler == null ? "RANDOM" : sampler,
        nTrials == null ? 12 : nTrials,
        kForMetrics == null ? 10 : kForMetrics,
        seed == null ? 0L : seed,
        querySampleSize == null ? 20 : querySampleSize,
        metadataField,
        mMin == null ? 8 : mMin,
        mMax == null ? 64 : mMax,
        efMin == null ? 50 : efMin,
        efMax == null ? 400 : efMax,
        recallWeight == null ? 1.0 : recallWeight,
        ndcgWeight == null ? 0.5 : ndcgWeight,
        latencyWeight == null ? 0.0 : latencyWeight);
  }
}
