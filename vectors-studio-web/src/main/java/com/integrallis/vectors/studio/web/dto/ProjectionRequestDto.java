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
package com.integrallis.vectors.studio.web.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import java.util.Map;

/**
 * JSON wire form of {@link com.integrallis.vectors.studio.core.projection.ProjectionRequest}. The
 * {@code params} map carries per-algorithm hyperparameters as a flat object whose keys mirror the
 * fields of {@link com.integrallis.vectors.studio.core.projection.ProjectionParams.PcaParams},
 * {@link com.integrallis.vectors.studio.core.projection.ProjectionParams.TsneParams}, or {@link
 * com.integrallis.vectors.studio.core.projection.ProjectionParams.UmapParams} depending on {@link
 * #algorithm()}; missing keys fall back to per-algorithm defaults. {@code sphereize} L2-normalises
 * each input row before fitting (TF Embedding Projector "Sphereize data").
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ProjectionRequestDto(
    String collection,
    ProjectionAlgorithm algorithm,
    int dimensions,
    int sampleSize,
    Map<String, Object> params,
    Boolean sphereize) {

  /**
   * Defensively copies {@code params} into an immutable map so the DTO is genuinely immutable: a
   * caller cannot mutate the map after construction, and the {@link #params()} accessor cannot leak
   * a mutable reference. A {@code null} map (field absent on the wire) is normalised to an empty
   * map.
   *
   * <p>{@link Map#copyOf} rejects {@code null} values; that is intentional here. The wire contract
   * is "omit a key to take its per-algorithm default" — a present key with a {@code null} value is
   * a malformed request and is rejected at the boundary rather than silently mishandled later.
   */
  public ProjectionRequestDto {
    params = params == null ? Map.of() : Map.copyOf(params);
  }
}
