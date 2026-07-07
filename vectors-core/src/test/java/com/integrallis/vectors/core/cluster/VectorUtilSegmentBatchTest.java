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
package com.integrallis.vectors.core.cluster;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Parity for the fused off-heap ({@link MemorySegment}) GEMV kernels ({@link
 * VectorUtil#batchDotProduct(float[], MemorySegment[], int, float[], int)} / {@link
 * VectorUtil#batchSquaredL2(float[], MemorySegment[], int, float[], int)}) against both the {@code
 * float[][]} fused kernels and the per-pair scalar reference. Correctness of these kernels
 * underpins the zero-copy HNSW scoring path, so results must match within float32 tolerance.
 */
@Tag("unit")
class VectorUtilSegmentBatchTest {

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

  /** Lays out {@code matrix} rows as individual off-heap little-endian float segments. */
  private static MemorySegment[] toSegments(Arena arena, float[][] matrix, int dim) {
    MemorySegment[] rows = new MemorySegment[matrix.length];
    for (int i = 0; i < matrix.length; i++) {
      MemorySegment seg = arena.allocate((long) dim * Float.BYTES);
      MemorySegment.copy(matrix[i], 0, seg, ValueLayout.JAVA_FLOAT, 0L, dim);
      rows[i] = seg;
    }
    return rows;
  }

  @ParameterizedTest(name = "dim={0}, count={1}")
  @CsvSource({
    "8,1", "8,3", "8,4", "8,7", "8,64",
    "64,1", "64,3", "64,4", "64,7", "64,64",
    "128,1", "128,3", "128,4", "128,7", "128,64",
    "768,1", "768,3", "768,4", "768,7", "768,64"
  })
  void batchDotProduct_segments_matchFloatArrayAndScalar(int dim, int count) {
    float[] query = randomVector(dim, 1_000L + dim * 31L + count);
    float[][] matrix = randomMatrix(count, dim, 2_000L + dim * 17L + count);

    float[] outArray = new float[count];
    VectorUtil.batchDotProduct(query, matrix, outArray, count);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, matrix, dim);
      float[] outSeg = new float[count];
      VectorUtil.batchDotProduct(query, rows, dim, outSeg, count);

      for (int i = 0; i < count; i++) {
        float scalar = VectorUtil.dotProduct(query, matrix[i]);
        assertThat(outSeg[i]).as("seg vs scalar row %d", i).isCloseTo(scalar, within(1e-4f));
        assertThat(outSeg[i])
            .as("seg vs float[][] row %d", i)
            .isCloseTo(outArray[i], within(1e-4f));
      }
    }
  }

  @ParameterizedTest(name = "dim={0}, count={1}")
  @CsvSource({
    "8,1", "8,3", "8,4", "8,7", "8,64",
    "64,1", "64,3", "64,4", "64,7", "64,64",
    "128,1", "128,3", "128,4", "128,7", "128,64",
    "768,1", "768,3", "768,4", "768,7", "768,64"
  })
  void batchSquaredL2_segments_matchFloatArrayAndScalar(int dim, int count) {
    float[] query = randomVector(dim, 3_000L + dim * 13L + count);
    float[][] matrix = randomMatrix(count, dim, 4_000L + dim * 19L + count);

    float[] outArray = new float[count];
    VectorUtil.batchSquaredL2(query, matrix, outArray, count);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, matrix, dim);
      float[] outSeg = new float[count];
      VectorUtil.batchSquaredL2(query, rows, dim, outSeg, count);

      for (int i = 0; i < count; i++) {
        float scalar = VectorUtil.squareDistance(query, matrix[i]);
        // L2 on 768-dim random vectors accumulates ~1.5e3 magnitude; allow proportional slack.
        float tol = Math.max(1e-3f, Math.abs(scalar) * 1e-5f);
        assertThat(outSeg[i]).as("seg vs scalar row %d", i).isCloseTo(scalar, within(tol));
        assertThat(outSeg[i]).as("seg vs float[][] row %d", i).isCloseTo(outArray[i], within(tol));
      }
    }
  }

  /** Odd dimensions exercise the SIMD-loop scalar tail inside the fused kernel. */
  @ParameterizedTest
  @ValueSource(ints = {1, 3, 5, 17, 33, 129, 257})
  void batchDotProduct_segments_oddDimension_matchesScalar(int dim) {
    int count = 32;
    float[] query = randomVector(dim, 50L);
    float[][] matrix = randomMatrix(count, dim, 51L);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, matrix, dim);
      float[] out = new float[count];
      VectorUtil.batchDotProduct(query, rows, dim, out, count);
      for (int i = 0; i < count; i++) {
        assertThat(out[i]).isCloseTo(VectorUtil.dotProduct(query, matrix[i]), within(1e-3f));
      }
    }
  }

  @Test
  void zeroCount_isNoOp() {
    float[] query = randomVector(8, 7L);
    VectorUtil.batchDotProduct(query, new MemorySegment[0], 8, new float[0], 0);
    VectorUtil.batchSquaredL2(query, new MemorySegment[0], 8, new float[0], 0);
  }

  @Test
  void validatesArguments() {
    float[] query = randomVector(4, 8L);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, randomMatrix(1, 4, 9L), 4);
      assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, rows, 4, new float[0], 1))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, rows, 4, new float[1], 2))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> VectorUtil.batchDotProduct(query, rows, 3, new float[1], 1))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> VectorUtil.batchDotProduct(null, rows, 4, new float[1], 1))
          .isInstanceOf(NullPointerException.class);
    }
  }
}
