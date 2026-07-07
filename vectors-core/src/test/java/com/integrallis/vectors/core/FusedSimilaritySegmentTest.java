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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Parity for {@link FusedSimilarity#bulkCompareSegments} against the on-heap {@link
 * FusedSimilarity#bulkCompare} and the per-row {@link SimilarityFunction#compare} reference, for
 * every {@link SimilarityFunction}.
 */
@Tag("unit")
class FusedSimilaritySegmentTest {

  private static final int DIM = 96;
  private static final int COUNT = 37; // full 4-row groups + a 1-row tail

  private static float[][] randomMatrix(int rows, int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[][] m = new float[rows][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private static float[] randomVector(int dim, long seed) {
    java.util.Random rng = new java.util.Random(seed);
    float[] v = new float[dim];
    for (int d = 0; d < dim; d++) v[d] = rng.nextFloat() * 2f - 1f;
    return v;
  }

  @ParameterizedTest
  @EnumSource(SimilarityFunction.class)
  void bulkCompareSegments_matchesFloatArrayAndScalar(SimilarityFunction sim) {
    float[] query = randomVector(DIM, 11L);
    float[][] pool = randomMatrix(COUNT, DIM, 22L);

    float[] scratchA = new float[COUNT];
    float[] scoresArray = new float[COUNT];
    FusedSimilarity.bulkCompare(sim, query, pool, scratchA, scoresArray, COUNT);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = new MemorySegment[COUNT];
      for (int i = 0; i < COUNT; i++) {
        MemorySegment seg = arena.allocate((long) DIM * Float.BYTES);
        MemorySegment.copy(pool[i], 0, seg, ValueLayout.JAVA_FLOAT, 0L, DIM);
        rows[i] = seg;
      }
      float[] scratchB = new float[COUNT];
      float[] scoresSeg = new float[COUNT];
      FusedSimilarity.bulkCompareSegments(sim, query, rows, DIM, scratchB, scoresSeg, COUNT);

      Offset<Float> tol = Offset.offset(1e-4f);
      for (int i = 0; i < COUNT; i++) {
        assertThat(scoresSeg[i])
            .as("%s seg vs scalar row %d", sim, i)
            .isCloseTo(sim.compare(query, pool[i]), tol);
        assertThat(scoresSeg[i])
            .as("%s seg vs float[][] row %d", sim, i)
            .isCloseTo(scoresArray[i], tol);
      }
    }
  }

  @ParameterizedTest
  @EnumSource(SimilarityFunction.class)
  void zeroCount_isNoOp(SimilarityFunction sim) {
    FusedSimilarity.bulkCompareSegments(
        sim, randomVector(DIM, 1L), new MemorySegment[0], DIM, new float[0], new float[0], 0);
  }

  @ParameterizedTest
  @EnumSource(SimilarityFunction.class)
  void validatesArguments(SimilarityFunction sim) {
    float[] query = randomVector(DIM, 1L);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = new MemorySegment[1];
      MemorySegment seg = arena.allocate((long) DIM * Float.BYTES);
      MemorySegment.copy(query, 0, seg, ValueLayout.JAVA_FLOAT, 0L, DIM);
      rows[0] = seg;
      assertThatThrownBy(
              () ->
                  FusedSimilarity.bulkCompareSegments(
                      null, query, rows, DIM, new float[1], new float[1], 1))
          .isInstanceOf(NullPointerException.class);
      assertThatThrownBy(
              () ->
                  FusedSimilarity.bulkCompareSegments(
                      sim, query, rows, DIM, new float[1], new float[1], 2))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  FusedSimilarity.bulkCompareSegments(
                      sim, query, rows, DIM, new float[0], new float[1], 1))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(
              () ->
                  FusedSimilarity.bulkCompareSegments(
                      sim, query, rows, DIM, new float[1], new float[0], 1))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
