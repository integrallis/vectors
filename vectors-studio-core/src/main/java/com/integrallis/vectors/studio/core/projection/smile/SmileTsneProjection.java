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

/** t-SNE projection backed by Smile's {@code smile.manifold.TSNE}. */
public final class SmileTsneProjection implements Projection {

  private final TsneParams params;
  private final int dimensions;

  public SmileTsneProjection(TsneParams params, int dimensions) {
    this.params = params;
    this.dimensions = dimensions;
  }

  @Override
  @SuppressWarnings("try") // try-with-resources holds heartbeat lifetime, body intentionally
  // doesn't reference it
  public ProjectionResult run(float[][] data, ProgressListener listener) {
    long start = System.currentTimeMillis();
    SmilePcaProjection.checkInterrupted();
    double[][] dd = SmilePcaProjection.toDouble(data);
    SmilePcaProjection.checkInterrupted();
    TSNE.Options opts =
        new TSNE.Options(
            dimensions, params.perplexity(), params.learningRate(), 12.0, params.iterations());
    // Smile's TSNE.fit is a blocking call with no mid-iteration callbacks. The heartbeat emits
    // time-based progress estimates with coords == null so consumers can render "still computing"
    // honestly rather than the previous 0%→100% synthetic jump (audit T4.12).
    TSNE result;
    try (SmileProgressHeartbeat ignored =
        SmileProgressHeartbeat.start(listener, params.iterations())) {
      result = TSNE.fit(dd, opts);
    }
    SmilePcaProjection.checkInterrupted();
    float[][] coords = SmilePcaProjection.toFloat(result.coordinates());
    long ms = System.currentTimeMillis() - start;
    ProjectionResult out =
        new ProjectionResult(coords, ProjectionAlgorithm.TSNE, params, ms, null, null);
    if (listener != null) {
      // Terminal call (iter == total) always carries real coordinates.
      listener.onIteration(params.iterations(), params.iterations(), coords);
      listener.onDone(out);
    }
    return out;
  }

  @Override
  public ProjectionAlgorithm algorithm() {
    return ProjectionAlgorithm.TSNE;
  }
}
