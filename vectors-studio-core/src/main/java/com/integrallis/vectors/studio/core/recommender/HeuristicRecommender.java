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
 * Pure-function recommender that maps {@link DatasetStats} to a {@link ProjectionRecommendation}.
 */
public final class HeuristicRecommender {

  /** Returns a sensible starting projection for the supplied dataset and target dimensionality. */
  public ProjectionRecommendation recommend(DatasetStats s, int targetDims) {
    if (s.dim() > 4 * s.n() || s.sparsityRatio() > 0.8) {
      return pca(targetDims, "d≫N or sparse: PCA stabilises before any nonlinear DR");
    }
    if (s.n() < 5_000) {
      int perp = Math.max(5, Math.min(50, s.n() / 100));
      return new ProjectionRecommendation(
          ProjectionAlgorithm.TSNE,
          new ProjectionParams.TsneParams(perp, 200.0, 1000, 42L),
          "N<5k: t-SNE preserves local cluster structure best at this scale",
          null);
    }
    if (s.n() <= 100_000) {
      return new ProjectionRecommendation(
          ProjectionAlgorithm.UMAP,
          new ProjectionParams.UmapParams(15, 0.1, 200, 42L),
          "5k≤N≤100k: UMAP balances global+local structure with linear scaling",
          null);
    }
    return pca(50, "N>100k: start with PCA-50 for downstream UMAP composition");
  }

  private ProjectionRecommendation pca(int dims, String rationale) {
    return new ProjectionRecommendation(
        ProjectionAlgorithm.PCA,
        new ProjectionParams.PcaParams(dims, true, false),
        rationale,
        null);
  }
}
