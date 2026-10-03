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
package com.integrallis.vectors.db.index;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PreparedFlatScanTest {
  @Test
  void singleAndBatchPreserveScoresAndTiesAndRecomputeAfterQueryMutation() {
    var random = new SplittableRandom(7932);
    for (int dim : new int[] {7, 32, 129, 512, 768}) {
      float[][] rows = new float[256][dim];
      float[][] queries = new float[3][dim];
      for (var matrix : new float[][][] {rows, queries})
        for (var row : matrix)
          for (int i = 0; i < dim; i++) row[i] = (float) random.nextDouble(-1, 1);
      // Identical rows exercise tied scores at the top-k boundary.
      for (int i = 0; i < 10; i++) rows[i] = queries[0].clone();
      var index = new FlatScanAdapter();
      for (var metric : SimilarityFunction.values()) {
        index.build(rows, metric);
        for (int k : new int[] {1, 10, 256, 300}) {
          var batch = index.searchBatch(queries, k, 128, 1f);
          for (int qi = 0; qi < queries.length; qi++) {
            var single = index.search(queries[qi], k, 128, 1f);
            assertArrayEquals(single.ordinals(), batch[qi].ordinals());
            // Cold Vector API reductions can differ by an ULP while the JVM changes tiers.
            // Exact compiled kernel parity is checked by PreparedCosineTest's isolated probe,
            // and the revision benchmark compares the complete warmed result bits.
            assertArrayEquals(single.scores(), batch[qi].scores(), 1e-6f);
            assertEquals(Math.min(k, rows.length), single.ordinals().length);
            for (int i = 0; i < single.ordinals().length; i++) {
              assertEquals(
                  metric.compare(queries[qi], rows[single.ordinals()[i]]),
                  single.scores()[i],
                  1e-6f);
              if (i > 0) assertTrue(single.scores()[i - 1] >= single.scores()[i]);
            }
          }
        }
      }
      index.build(rows, SimilarityFunction.COSINE);
      index.search(queries[0], 10, 128, 1f);
      Arrays.fill(queries[0], .125f);
      var result = index.search(queries[0], 10, 128, 1f);
      for (int i = 0; i < result.ordinals().length; i++)
        assertEquals(
            SimilarityFunction.COSINE.compare(queries[0], rows[result.ordinals()[i]]),
            result.scores()[i],
            1e-6f);
    }
  }
}
