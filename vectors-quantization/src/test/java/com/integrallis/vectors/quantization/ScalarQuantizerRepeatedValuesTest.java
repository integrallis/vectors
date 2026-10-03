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
package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.*;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Tag("unit")
class ScalarQuantizerRepeatedValuesTest {
  @Test
  @Timeout(10)
  void repeatedValuesMatchSortedQuantilesWithoutQuadraticWork() {
    float[][] rows = new float[2000][128];
    Random random = new Random(42);
    float[] sorted = new float[2000 * 128];
    int at = 0;
    for (float[] row : rows)
      for (int d = 0; d < row.length; d++) {
        row[d] = random.nextInt(256); // duplicate-rich, as in image data
        sorted[at++] = row[d];
      }
    Arrays.sort(sorted);
    for (float ci : new float[] {.8f, .99f, 1f}) {
      float[] actual = ScalarQuantizer.computeQuantiles(new ArrayVectorDataset(rows), ci);
      int k = (int) (sorted.length * (1f - ci) / 2f + .5f);
      assertThat(actual).containsExactly(sorted[k], sorted[sorted.length - 1 - k]);
    }
    for (float[] row : rows) Arrays.fill(row, 0f);
    float[] bounds = ScalarQuantizer.computeQuantiles(new ArrayVectorDataset(rows), .99f);
    assertThat(bounds[0]).isLessThan(0f);
    assertThat(bounds[1]).isGreaterThan(0f);
  }
}
