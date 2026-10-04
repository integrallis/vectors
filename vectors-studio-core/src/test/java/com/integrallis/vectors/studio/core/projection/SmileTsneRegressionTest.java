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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.studio.core.projection.smile.SmileTsneProjection;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class SmileTsneRegressionTest {
  @Test
  void normalizedEmbeddingsDoNotStopAtTheCollapsedEarlyExaggerationState() {
    float[][] data = normalizedClusters();
    var p = new SmileTsneProjection(new ProjectionParams.TsneParams(30, 70, 500, 42), 2);
    float[][] coords = p.run(data, null).coords();
    double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
    for (float[] point : coords) {
      for (float value : point) assertThat(Float.isFinite(value)).isTrue();
      min = Math.min(min, point[0]);
      max = Math.max(max, point[0]);
    }
    assertThat(max - min)
        .as("projection must escape its near-zero initialization")
        .isGreaterThan(0.01);
  }

  @Test
  void collapsedCoordinatesAreReportedAsAnErrorInsteadOfSuccess() {
    var p = new SmileTsneProjection(new ProjectionParams.TsneParams(30, 40, 500, 42), 2);
    assertThatThrownBy(() -> p.run(normalizedClusters(), null))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("collapsed to a point");
  }

  private static float[][] normalizedClusters() {
    Random rng = new Random(7);
    float[][] data = new float[600][32];
    for (int i = 0; i < data.length; i++) {
      float[] row = data[i];
      double norm = 0;
      for (int j = 0; j < row.length; j++) {
        row[j] = (float) (rng.nextGaussian() * 0.25 + (j == i % 14 ? 1 : 0));
        norm += row[j] * row[j];
      }
      for (int j = 0; j < row.length; j++) row[j] /= (float) Math.sqrt(norm);
    }
    return data;
  }

  @Test
  void seedReproducesCoordinates() {
    float[][] data = SmilePcaProjectionTest.ThreeClusterData.generate(60, 16, 7L);
    var p = new SmileTsneProjection(new ProjectionParams.TsneParams(10, 200, 300, 42), 2);
    var first = p.run(data, null).coords();
    var second = p.run(data, null).coords();
    for (int i = 0; i < first.length; i++) assertThat(second[i]).containsExactly(first[i]);
  }
}
