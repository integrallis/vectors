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
package com.integrallis.vectors.studio.core.recommender;

import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionParams;

/**
 * Recommended starting point for a projection job.
 *
 * @param algorithm chosen algorithm
 * @param params chosen hyperparameters
 * @param rationale heuristic explanation
 * @param llmExplanation optional LLM-generated narrative; null when not enriched
 */
public record ProjectionRecommendation(
    ProjectionAlgorithm algorithm,
    ProjectionParams params,
    String rationale,
    String llmExplanation) {}
