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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.studio.core.projection.smile.SmilePcaProjection;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SmilePcaProjectionTest {

  @Test
  void projects3ClusterDataInto2dWithMostVarianceCaptured() {
    float[][] data = ThreeClusterData.generate(150, 16, 12345L);
    SmilePcaProjection p =
        new SmilePcaProjection(new ProjectionParams.PcaParams(2, true, false), 2);
    ProjectionResult r = p.run(data, ProgressListener.noop());

    assertThat(r.algorithm()).isEqualTo(ProjectionAlgorithm.PCA);
    assertThat(r.coords()).hasNumberOfRows(150);
    assertThat(r.coords()[0]).hasSize(2);
    assertThat(r.varianceExplained()).isNotNull().hasSize(2);
    double sum = r.varianceExplained()[0] + r.varianceExplained()[1];
    assertThat(sum).isBetween(0.20, 1.0);
  }

  @Test
  void queryProjectorReprojectsTrainingRowToItsInSampleCoords() {
    float[][] data = ThreeClusterData.generate(120, 16, 999L);
    SmilePcaProjection p =
        new SmilePcaProjection(new ProjectionParams.PcaParams(2, true, false), 2);
    ProjectionResult r = p.run(data, ProgressListener.noop());

    QueryProjector qp = r.queryProjector();
    assertThat(qp).isNotNull();

    // PCA is a linear out-of-sample transform: projecting a training row through the retained
    // projector must reproduce that row's in-sample coordinates (same fitted PCA, same frame).
    int row = 42;
    float[] projected = qp.project(data[row]);
    assertThat(projected).hasSize(2);
    assertThat((double) projected[0]).isCloseTo(r.coords()[row][0], within(1e-3));
    assertThat((double) projected[1]).isCloseTo(r.coords()[row][1], within(1e-3));

    // A novel (non-training) vector projects to finite coords of the right dimension.
    float[] novel = new float[16];
    for (int i = 0; i < novel.length; i++) novel[i] = (float) (Math.sin(i) * 3.0);
    float[] novelCoords = qp.project(novel);
    assertThat(novelCoords).hasSize(2);
    assertThat(Float.isFinite(novelCoords[0])).isTrue();
    assertThat(Float.isFinite(novelCoords[1])).isTrue();
  }

  @Test
  void invokesProgressListenerOnceAndDone() {
    float[][] data = ThreeClusterData.generate(60, 8, 0L);
    int[] iterations = {0};
    boolean[] done = {false};
    new SmilePcaProjection(new ProjectionParams.PcaParams(2, true, false), 2)
        .run(
            data,
            new ProgressListener() {
              @Override
              public void onIteration(int iter, int total, float[][] coords) {
                iterations[0]++;
              }

              @Override
              public void onDone(ProjectionResult result) {
                done[0] = true;
              }
            });
    assertThat(iterations[0]).isEqualTo(1);
    assertThat(done[0]).isTrue();
  }

  /** Generates 3 well-separated Gaussian clusters in {@code dim} dimensions. */
  static final class ThreeClusterData {
    private ThreeClusterData() {}

    static float[][] generate(int n, int dim, long seed) {
      Random rng = new Random(seed);
      float[][] data = new float[n][dim];
      for (int i = 0; i < n; i++) {
        int cluster = i % 3;
        for (int j = 0; j < dim; j++) {
          float center = cluster == 0 ? -8f : (cluster == 1 ? 0f : 8f);
          data[i][j] = (float) (center + rng.nextGaussian() * 0.5);
        }
      }
      return data;
    }
  }
}
