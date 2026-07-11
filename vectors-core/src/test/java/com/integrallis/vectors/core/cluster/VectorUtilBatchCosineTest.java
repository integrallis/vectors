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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Parity for the fused batch cosine kernels ({@link VectorUtil#batchCosine(float[], float[][],
 * float[], int)} and {@link VectorUtil#batchCosine(float[], MemorySegment[], int, float[], int)})
 * against the per-row {@link VectorUtil#cosine(float[], float[])} reference. These kernels back the
 * non-normalized / preserve-original-vectors cosine HNSW scoring path, so their raw cosine values
 * must match the per-row kernel within float32 tolerance.
 */
@Tag("unit")
class VectorUtilBatchCosineTest {

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

  private static MemorySegment[] toSegments(Arena arena, float[][] matrix, int dim) {
    MemorySegment[] rows = new MemorySegment[matrix.length];
    for (int i = 0; i < matrix.length; i++) {
      MemorySegment seg = arena.allocate((long) dim * Float.BYTES);
      MemorySegment.copy(matrix[i], 0, seg, ValueLayout.JAVA_FLOAT, 0L, dim);
      rows[i] = seg;
    }
    return rows;
  }

  // Relative tolerance: cosine of 768-dim random vectors involves large dot/norm reductions whose
  // last-ULP fp order differs between the fused (query-norm-once) kernel and the per-row reference.
  private static float tol(int dim) {
    return dim >= 512 ? 1e-4f : 1e-5f;
  }

  @ParameterizedTest(name = "dim={0}, count={1}")
  @CsvSource({
    "8,1", "8,3", "8,4", "8,7", "8,64",
    "64,1", "64,3", "64,4", "64,7", "64,64",
    "128,1", "128,3", "128,4", "128,7", "128,64",
    "768,1", "768,3", "768,4", "768,7", "768,64"
  })
  void batchCosine_floatArray_matchesPerRowCosine(int dim, int count) {
    float[] query = randomVector(dim, 1_000L + dim * 31L + count);
    float[][] matrix = randomMatrix(count, dim, 2_000L + dim * 17L + count);

    float[] out = new float[count];
    VectorUtil.batchCosine(query, matrix, out, count);

    for (int i = 0; i < count; i++) {
      float expected = VectorUtil.cosine(query, matrix[i]);
      assertThat(out[i])
          .as("float[][] cosine row %d (dim=%d)", i, dim)
          .isCloseTo(expected, within(tol(dim)));
    }
  }

  @ParameterizedTest(name = "dim={0}, count={1}")
  @CsvSource({
    "8,1", "8,3", "8,4", "8,7", "8,64",
    "64,1", "64,3", "64,4", "64,7", "64,64",
    "128,1", "128,3", "128,4", "128,7", "128,64",
    "768,1", "768,3", "768,4", "768,7", "768,64"
  })
  void batchCosine_segments_matchPerRowCosineAndFloatArray(int dim, int count) {
    float[] query = randomVector(dim, 3_000L + dim * 13L + count);
    float[][] matrix = randomMatrix(count, dim, 4_000L + dim * 19L + count);

    float[] outArray = new float[count];
    VectorUtil.batchCosine(query, matrix, outArray, count);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, matrix, dim);
      float[] outSeg = new float[count];
      VectorUtil.batchCosine(query, rows, dim, outSeg, count);

      for (int i = 0; i < count; i++) {
        float expected = VectorUtil.cosine(query, matrix[i]);
        assertThat(outSeg[i])
            .as("seg vs per-row row %d (dim=%d)", i, dim)
            .isCloseTo(expected, within(tol(dim)));
        assertThat(outSeg[i])
            .as("seg vs float[][] row %d (dim=%d)", i, dim)
            .isCloseTo(outArray[i], within(tol(dim)));
      }
    }
  }

  /**
   * A zero-vector row makes the row norm zero, so the per-row cosine computes {@code 0/0 = NaN}.
   * The fused kernels (both float[][] and segment) must reproduce that exact value — a divergence
   * here would mean the fused path scores an unnormalizable vector differently from the reference.
   */
  @Test
  void batchCosine_zeroRow_matchesPerRowNaN() {
    int dim = 128;
    int count = 5;
    float[] query = randomVector(dim, 77L);
    float[][] matrix = randomMatrix(count, dim, 88L);
    int zeroRow = 2;
    java.util.Arrays.fill(matrix[zeroRow], 0f);

    float perRow = VectorUtil.cosine(query, matrix[zeroRow]);
    assertThat(perRow).as("per-row cosine of a zero row is NaN").isNaN();

    float[] outArray = new float[count];
    VectorUtil.batchCosine(query, matrix, outArray, count);
    assertThat(outArray[zeroRow]).as("float[][] fused cosine of a zero row").isNaN();

    // Non-zero rows still match the per-row reference.
    for (int i = 0; i < count; i++) {
      if (i == zeroRow) continue;
      assertThat(outArray[i]).isCloseTo(VectorUtil.cosine(query, matrix[i]), within(1e-5f));
    }

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment[] rows = toSegments(arena, matrix, dim);
      float[] outSeg = new float[count];
      VectorUtil.batchCosine(query, rows, dim, outSeg, count);
      assertThat(outSeg[zeroRow]).as("segment fused cosine of a zero row").isNaN();
      for (int i = 0; i < count; i++) {
        if (i == zeroRow) continue;
        assertThat(outSeg[i]).isCloseTo(VectorUtil.cosine(query, matrix[i]), within(1e-5f));
      }
    }
  }
}
