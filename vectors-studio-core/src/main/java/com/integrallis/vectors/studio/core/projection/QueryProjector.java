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
 * Projects a single (possibly out-of-sample) vector into a previously fitted projection's
 * low-dimensional coordinate frame — used to place a search query as its own point in the same
 * space as the rendered cloud.
 *
 * <p>Only linear projections (PCA) can offer an exact out-of-sample transform. Non-linear
 * algorithms (t-SNE, UMAP) expose no such transform, so their {@link ProjectionResult} carries a
 * {@code null} projector and the client approximates the query position client-side.
 */
@FunctionalInterface
public interface QueryProjector {

  /**
   * Projects {@code vector} into the fitted space, returning coordinates of the fit's dimension.
   */
  float[] project(float[] vector);
}
