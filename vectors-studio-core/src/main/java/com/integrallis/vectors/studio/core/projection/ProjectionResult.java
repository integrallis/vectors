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
package com.integrallis.vectors.studio.core.projection;

/**
 * Output of a projection job.
 *
 * @param coords {@code [N][dimensions]} matrix of low-dim coordinates
 * @param algorithm the algorithm used
 * @param params the algorithm parameters used
 * @param durationMs wall-clock fit time
 * @param varianceExplained PCA only — fraction of variance retained per component; null for others
 * @param queryProjector out-of-sample projector for placing a query point; non-null for PCA
 *     (linear, exact), null for t-SNE/UMAP (no transform — the client approximates instead)
 */
public record ProjectionResult(
    float[][] coords,
    ProjectionAlgorithm algorithm,
    ProjectionParams params,
    long durationMs,
    double[] varianceExplained,
    QueryProjector queryProjector) {}
