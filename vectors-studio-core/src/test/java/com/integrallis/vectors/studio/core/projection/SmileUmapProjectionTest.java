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

import com.integrallis.vectors.studio.core.projection.smile.SmileUmapProjection;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class SmileUmapProjectionTest {

  @Test
  void embeds3ClusterDataIntoTwoDimensions() {
    float[][] data = SmilePcaProjectionTest.ThreeClusterData.generate(120, 16, 9L);
    int[] iters = {0};
    boolean[] done = {false};
    SmileUmapProjection p =
        new SmileUmapProjection(new ProjectionParams.UmapParams(10, 0.1, 100, 42L), 2);
    ProjectionResult r =
        p.run(
            data,
            new ProgressListener() {
              @Override
              public void onIteration(int iter, int total, float[][] coords) {
                iters[0]++;
                assertThat(iter).isEqualTo(total);
                assertThat(coords).hasNumberOfRows(120);
                assertThat(coords[0]).hasSize(2);
              }

              @Override
              public void onDone(ProjectionResult result) {
                done[0] = true;
              }
            });
    assertThat(r.algorithm()).isEqualTo(ProjectionAlgorithm.UMAP);
    assertThat(r.coords()).hasNumberOfRows(120);
    assertThat(r.coords()[0]).hasSize(2);
    assertThat(iters[0]).isEqualTo(1);
    assertThat(done[0]).isTrue();
  }
}
