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
import java.util.Objects;

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

  /**
   * Defaults applied when a field is null on the wire.
   *
   * <p>Uses {@link Objects#requireNonNullElse} rather than {@code x == null ? literal : x}: in a
   * ternary, a boxed field and a primitive literal force the boxed branch to unbox and the result
   * to rebox ({@code BX_UNBOXING_IMMEDIATELY_REBOXED}). {@code requireNonNullElse} keeps both
   * operands boxed, so the non-null field passes through with no unbox/rebox round-trip.
   */
  public OptimizeRequestDto withDefaults() {
    return new OptimizeRequestDto(
        collection,
        Objects.requireNonNullElse(sampler, "RANDOM"),
        Objects.requireNonNullElse(nTrials, 12),
        Objects.requireNonNullElse(kForMetrics, 10),
        Objects.requireNonNullElse(seed, 0L),
        Objects.requireNonNullElse(querySampleSize, 20),
        metadataField,
        Objects.requireNonNullElse(mMin, 8),
        Objects.requireNonNullElse(mMax, 64),
        Objects.requireNonNullElse(efMin, 50),
        Objects.requireNonNullElse(efMax, 400),
        Objects.requireNonNullElse(recallWeight, 1.0),
        Objects.requireNonNullElse(ndcgWeight, 0.5),
        Objects.requireNonNullElse(latencyWeight, 0.0));
  }
}
