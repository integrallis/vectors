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
package com.integrallis.vectors.studio.core.projection.smile;

import com.integrallis.vectors.studio.core.projection.ProgressListener;
import com.integrallis.vectors.studio.core.projection.Projection;
import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionParams.UmapParams;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import smile.manifold.UMAP;

/** UMAP projection backed by Smile's {@code smile.manifold.UMAP}. */
public final class SmileUmapProjection implements Projection {

  private final UmapParams params;
  private final int dimensions;

  public SmileUmapProjection(UmapParams params, int dimensions) {
    this.params = params;
    this.dimensions = dimensions;
  }

  @Override
  public ProjectionResult run(float[][] data, ProgressListener listener) {
    long start = System.currentTimeMillis();
    double[][] dd = SmilePcaProjection.toDouble(data);
    UMAP.Options opts =
        new UMAP.Options(
            params.neighbors(),
            dimensions,
            params.iterations(),
            1.0,
            params.minDist(),
            1.0,
            5,
            1.0,
            1.0);
    double[][] projected = UMAP.fit(dd, opts);
    float[][] coords = SmilePcaProjection.toFloat(projected);
    long ms = System.currentTimeMillis() - start;
    ProjectionResult out = new ProjectionResult(coords, ProjectionAlgorithm.UMAP, params, ms, null);
    if (listener != null) {
      listener.onIteration(params.iterations(), params.iterations(), coords);
      listener.onDone(out);
    }
    return out;
  }

  @Override
  public ProjectionAlgorithm algorithm() {
    return ProjectionAlgorithm.UMAP;
  }
}
