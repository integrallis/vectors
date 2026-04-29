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

import com.integrallis.vectors.studio.core.projection.ProjectionParams.PcaParams;
import com.integrallis.vectors.studio.core.projection.ProjectionParams.TsneParams;
import com.integrallis.vectors.studio.core.projection.ProjectionParams.UmapParams;
import com.integrallis.vectors.studio.core.projection.smile.SmilePcaProjection;
import com.integrallis.vectors.studio.core.projection.smile.SmileTsneProjection;
import com.integrallis.vectors.studio.core.projection.smile.SmileUmapProjection;

/** Dispatches a {@link ProjectionRequest} to the matching {@link Projection} implementation. */
public final class ProjectionRunner {

  /** Runs the projection synchronously, forwarding progress to {@code listener}. */
  public ProjectionResult run(ProjectionRequest req, float[][] data, ProgressListener listener) {
    Projection p =
        switch (req.algorithm()) {
          case PCA -> new SmilePcaProjection((PcaParams) req.params(), req.dimensions());
          case TSNE -> new SmileTsneProjection((TsneParams) req.params(), req.dimensions());
          case UMAP -> new SmileUmapProjection((UmapParams) req.params(), req.dimensions());
        };
    return p.run(data, listener);
  }
}
