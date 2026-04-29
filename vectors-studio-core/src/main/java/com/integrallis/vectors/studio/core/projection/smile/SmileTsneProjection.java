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
import com.integrallis.vectors.studio.core.projection.ProjectionParams.TsneParams;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import smile.manifold.TSNE;

/**
 * t-SNE projection backed by Smile's {@code smile.manifold.TSNE}.
 *
 * <p>Smile 6.0.0 exposes t-SNE as a one-shot {@code fit}; we report start/end iteration markers via
 * the listener so the UI sees at least two progress updates.
 */
public final class SmileTsneProjection implements Projection {

  private final TsneParams params;
  private final int dimensions;

  public SmileTsneProjection(TsneParams params, int dimensions) {
    this.params = params;
    this.dimensions = dimensions;
  }

  @Override
  public ProjectionResult run(float[][] data, ProgressListener listener) {
    long start = System.currentTimeMillis();
    double[][] dd = SmilePcaProjection.toDouble(data);
    if (listener != null) {
      listener.onIteration(0, params.iterations(), zeros(data.length));
    }
    TSNE.Options opts =
        new TSNE.Options(
            dimensions, params.perplexity(), params.learningRate(), 12.0, params.iterations());
    TSNE result = TSNE.fit(dd, opts);
    float[][] coords = SmilePcaProjection.toFloat(result.coordinates());
    long ms = System.currentTimeMillis() - start;
    ProjectionResult out = new ProjectionResult(coords, ProjectionAlgorithm.TSNE, params, ms, null);
    if (listener != null) {
      listener.onIteration(params.iterations() / 2, params.iterations(), coords);
      listener.onIteration(params.iterations(), params.iterations(), coords);
      listener.onDone(out);
    }
    return out;
  }

  @Override
  public ProjectionAlgorithm algorithm() {
    return ProjectionAlgorithm.TSNE;
  }

  private float[][] zeros(int n) {
    float[][] z = new float[n][dimensions];
    return z;
  }
}
