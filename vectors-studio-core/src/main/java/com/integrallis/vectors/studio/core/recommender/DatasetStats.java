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

/**
 * Summary statistics describing a dataset's structure, used by {@link HeuristicRecommender}.
 *
 * @param n number of samples
 * @param dim ambient dimension
 * @param intrinsicDimEstimate elbow estimate from PCA scree on a sample
 * @param labelCardinality number of distinct labels (null when unlabeled)
 * @param hasText whether the documents carry text bodies
 * @param sparsityRatio fraction of zero entries (TF-IDF detector)
 */
public record DatasetStats(
    int n,
    int dim,
    double intrinsicDimEstimate,
    Integer labelCardinality,
    boolean hasText,
    double sparsityRatio) {}
