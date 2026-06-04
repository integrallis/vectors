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
import com.integrallis.vectors.studio.core.projection.ProjectionParams.PcaParams;
import com.integrallis.vectors.studio.core.projection.ProjectionResult;
import java.util.concurrent.CancellationException;
import smile.feature.extraction.PCA;
import smile.tensor.Vector;

/** PCA projection backed by Smile's {@code smile.feature.extraction.PCA}. */
public final class SmilePcaProjection implements Projection {

  private final PcaParams params;
  private final int dimensions;

  public SmilePcaProjection(PcaParams params, int dimensions) {
    this.params = params;
    this.dimensions = dimensions;
  }

  @Override
  public ProjectionResult run(float[][] data, ProgressListener listener) {
    long start = System.currentTimeMillis();
    checkInterrupted();
    double[][] dd = toDouble(data);
    checkInterrupted();
    PCA full = PCA.fit(dd);
    checkInterrupted();
    PCA reduced = full.getProjection(dimensions);
    double[][] projected = reduced.apply(dd);
    checkInterrupted();
    float[][] coords = toFloat(projected);
    Vector vp = full.varianceProportion();
    double[] variance = new double[Math.min(vp.size(), dimensions)];
    for (int i = 0; i < variance.length; i++) variance[i] = vp.get(i);
    long ms = System.currentTimeMillis() - start;
    ProjectionResult result =
        new ProjectionResult(coords, ProjectionAlgorithm.PCA, params, ms, variance);
    if (listener != null) {
      listener.onIteration(1, 1, coords);
      listener.onDone(result);
    }
    return result;
  }

  @Override
  public ProjectionAlgorithm algorithm() {
    return ProjectionAlgorithm.PCA;
  }

  static double[][] toDouble(float[][] in) {
    double[][] out = new double[in.length][];
    for (int i = 0; i < in.length; i++) {
      checkInterrupted();
      double[] row = new double[in[i].length];
      for (int j = 0; j < in[i].length; j++) row[j] = in[i][j];
      out[i] = row;
    }
    return out;
  }

  static float[][] toFloat(double[][] in) {
    float[][] out = new float[in.length][];
    for (int i = 0; i < in.length; i++) {
      checkInterrupted();
      float[] row = new float[in[i].length];
      for (int j = 0; j < in[i].length; j++) row[j] = (float) in[i][j];
      out[i] = row;
    }
    return out;
  }

  static void checkInterrupted() {
    if (Thread.currentThread().isInterrupted()) {
      throw new CancellationException("projection interrupted");
    }
  }
}
