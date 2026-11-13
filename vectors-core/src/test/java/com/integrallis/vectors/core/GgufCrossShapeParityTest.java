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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Cross-shape arithmetic parity for grouped quantized batched matmul.
 *
 * <p>Issue #119 deliverable: verify that batched grouped projection (Q6_K, Q8_0) produces the same
 * results as N independent single-row projections, for batch sizes 1/2/4/8 and non-power-of-two
 * tails.
 */
@Tag("unit")
class GgufCrossShapeParityTest {

  private static final int[] BATCH_SIZES = {1, 2, 4, 8};
  private static final int[] TAIL_SIZES = {3, 5, 7, 9};

  // Representative panel shapes. Must be a multiple of K-quant block size (256).
  private static final int ROWS = 512;
  private static final int COLS = 2048;

  // Activation quantization scratch: Q8_K for Q6_K path, Q8_0 for Q8_0 path.
  private static final int Q8_K_BLOCK = 256; // elements per Q8_K super-block
  private static final int Q8_0_BLOCK = 32; // elements per Q8_0 block

  // Tolerance: activation quantization error is approximately 1% relative.
  // We allow 2% absolute for the parity check (generous, because independent vs batched
  // should be much tighter; this catches gross errors, not just quantization noise).
  private static final float TOLERANCE = 0.02f;

  // Fixed seeds so the test is deterministic.
  private static final long WEIGHT_SEED = 0xDEADBEEFCAFEBABEL;
  private static final long QUERY_SEED = 0xFEEDFACEFADEBABEL;

  @Test
  void q6k_batched_matches_independent_for_standard_batch_sizes() {
    Random rnd = new Random(WEIGHT_SEED);
    MemorySegment weights = createRandomQ6KWeights(rnd, ROWS, COLS);
    float[] outBatched = new float[8 * ROWS];
    float[] outIndependent = new float[8 * ROWS];

    for (int batchSize : new int[] {1, 2, 4, 8}) {
      Random qRnd = new Random(QUERY_SEED);
      float[] queries = createQueries(qRnd, batchSize, COLS);
      byte[] q8Quants = new byte[batchSize * COLS];
      float[] q8Scales = new float[batchSize * (COLS / Q8_K_BLOCK)];
      byte[] q8IndQuants = new byte[COLS];
      float[] q8IndScales = new float[COLS / Q8_K_BLOCK];

      // Batched path
      VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
          queries,
          weights,
          batchSize,
          ROWS,
          COLS,
          outBatched,
          q8Quants,
          q8Scales,
          GgufQ6BatchedKernel.ONE_QUERY_BLOCK);

      // Independent path: same queries, row by row
      for (int b = 0; b < batchSize; b++) {
        float[] singleQuery = new float[COLS];
        System.arraycopy(queries, b * COLS, singleQuery, 0, COLS);
        float[] singleOut = new float[ROWS];
        VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
            singleQuery, weights, ROWS, COLS, singleOut, q8IndQuants, q8IndScales);
        for (int r = 0; r < ROWS; r++) {
          outIndependent[b * ROWS + r] = singleOut[r];
        }
      }

      // Compare
      for (int i = 0; i < batchSize * ROWS; i++) {
        float expected = outIndependent[i];
        float actual = outBatched[i];
        assertThat(actual)
            .as("Q6_K batchSize=%d row=%d", batchSize, i)
            .isCloseTo(expected, within(Math.abs(expected) * TOLERANCE + TOLERANCE));
      }
    }
  }

  @Test
  void q6k_batched_matches_independent_for_tail_batch_sizes() {
    Random rnd = new Random(WEIGHT_SEED);
    MemorySegment weights = createRandomQ6KWeights(rnd, ROWS, COLS);

    for (int batchSize : TAIL_SIZES) {
      float[] outBatched = new float[batchSize * ROWS];
      float[] outIndependent = new float[batchSize * ROWS];
      Random qRnd = new Random(QUERY_SEED);
      float[] queries = createQueries(qRnd, batchSize, COLS);
      byte[] q8Quants = new byte[batchSize * COLS];
      float[] q8Scales = new float[batchSize * (COLS / Q8_K_BLOCK)];
      byte[] q8IndQuants = new byte[COLS];
      float[] q8IndScales = new float[COLS / Q8_K_BLOCK];

      VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
          queries,
          weights,
          batchSize,
          ROWS,
          COLS,
          outBatched,
          q8Quants,
          q8Scales,
          GgufQ6BatchedKernel.ONE_QUERY_BLOCK);

      for (int b = 0; b < batchSize; b++) {
        float[] singleQuery = new float[COLS];
        System.arraycopy(queries, b * COLS, singleQuery, 0, COLS);
        float[] singleOut = new float[ROWS];
        VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
            singleQuery, weights, ROWS, COLS, singleOut, q8IndQuants, q8IndScales);
        for (int r = 0; r < ROWS; r++) {
          outIndependent[b * ROWS + r] = singleOut[r];
        }
      }

      for (int i = 0; i < batchSize * ROWS; i++) {
        float expected = outIndependent[i];
        float actual = outBatched[i];
        assertThat(actual)
            .as("Q6_K tail batchSize=%d row=%d", batchSize, i)
            .isCloseTo(expected, within(Math.abs(expected) * TOLERANCE + TOLERANCE));
      }
    }
  }

  @Test
  void q8_0_batched_matches_independent_for_standard_batch_sizes() {
    Random rnd = new Random(WEIGHT_SEED);
    MemorySegment weights = createRandomQ8_0Weights(rnd, ROWS, COLS);
    float[] outBatched = new float[8 * ROWS];
    float[] outIndependent = new float[8 * ROWS];

    for (int batchSize : new int[] {1, 2, 4, 8}) {
      Random qRnd = new Random(QUERY_SEED);
      float[] queries = createQueries(qRnd, batchSize, COLS);
      byte[] q8Quants = new byte[batchSize * COLS];
      float[] q8Scales = new float[batchSize * (COLS / Q8_0_BLOCK)];
      byte[] q8IndQuants = new byte[COLS];
      float[] q8IndScales = new float[COLS / Q8_0_BLOCK];

      VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
          queries, weights, batchSize, ROWS, COLS, outBatched, q8Quants, q8Scales);

      for (int b = 0; b < batchSize; b++) {
        float[] singleQuery = new float[COLS];
        System.arraycopy(queries, b * COLS, singleQuery, 0, COLS);
        float[] singleOut = new float[ROWS];
        VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
            singleQuery, weights, ROWS, COLS, singleOut, q8IndQuants, q8IndScales);
        for (int r = 0; r < ROWS; r++) {
          outIndependent[b * ROWS + r] = singleOut[r];
        }
      }

      for (int i = 0; i < batchSize * ROWS; i++) {
        float expected = outIndependent[i];
        float actual = outBatched[i];
        assertThat(expected)
            .as("Q8_0 batchSize=%d row=%d expected is finite", batchSize, i)
            .isFinite();
        assertThat(actual).as("Q8_0 batchSize=%d row=%d actual is finite", batchSize, i).isFinite();
        float tol = Math.abs(expected) * TOLERANCE + TOLERANCE;
        assertThat(actual)
            .as("Q8_0 batchSize=%d row=%d", batchSize, i)
            .isCloseTo(expected, within(tol));
      }
    }
  }

  @Test
  void q8_0_batched_matches_independent_for_tail_batch_sizes() {
    Random rnd = new Random(WEIGHT_SEED);
    MemorySegment weights = createRandomQ8_0Weights(rnd, ROWS, COLS);

    for (int batchSize : TAIL_SIZES) {
      float[] outBatched = new float[batchSize * ROWS];
      float[] outIndependent = new float[batchSize * ROWS];
      Random qRnd = new Random(QUERY_SEED);
      float[] queries = createQueries(qRnd, batchSize, COLS);
      byte[] q8Quants = new byte[batchSize * COLS];
      float[] q8Scales = new float[batchSize * (COLS / Q8_0_BLOCK)];
      byte[] q8IndQuants = new byte[COLS];
      float[] q8IndScales = new float[COLS / Q8_0_BLOCK];

      VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
          queries, weights, batchSize, ROWS, COLS, outBatched, q8Quants, q8Scales);

      for (int b = 0; b < batchSize; b++) {
        float[] singleQuery = new float[COLS];
        System.arraycopy(queries, b * COLS, singleQuery, 0, COLS);
        float[] singleOut = new float[ROWS];
        VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
            singleQuery, weights, ROWS, COLS, singleOut, q8IndQuants, q8IndScales);
        for (int r = 0; r < ROWS; r++) {
          outIndependent[b * ROWS + r] = singleOut[r];
        }
      }

      for (int i = 0; i < batchSize * ROWS; i++) {
        float expected = outIndependent[i];
        float actual = outBatched[i];
        assertThat(expected)
            .as("Q8_0 tail batchSize=%d row=%d expected is finite", batchSize, i)
            .isFinite();
        assertThat(actual)
            .as("Q8_0 tail batchSize=%d row=%d actual is finite", batchSize, i)
            .isFinite();
        float tol = Math.abs(expected) * TOLERANCE + TOLERANCE;
        assertThat(actual)
            .as("Q8_0 tail batchSize=%d row=%d", batchSize, i)
            .isCloseTo(expected, within(tol));
      }
    }
  }

  // --- Helpers ---

  private static float[] createQueries(Random rnd, int batchSize, int cols) {
    float[] q = new float[batchSize * cols];
    for (int i = 0; i < q.length; i++) {
      q[i] = rnd.nextFloat() * 2.0f - 1.0f;
    }
    return q;
  }

  private static MemorySegment createRandomQ6KWeights(Random rnd, int rows, int cols) {
    // Each Q6_K block is 256 elements → 210 bytes
    int blocksPerRow = cols / 256;
    int blockBytes = 210;
    byte[] blocks = new byte[rows * blocksPerRow * blockBytes];
    rnd.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int offset = 0; offset < blocks.length; offset += blockBytes) {
      buffer.putShort(offset + 208, scale);
    }
    return MemorySegment.ofArray(blocks);
  }

  private static MemorySegment createRandomQ8_0Weights(Random rnd, int rows, int cols) {
    // Each Q8_0 block is 32 elements: [scale: 2 bytes][weights: 32 bytes] = 34 bytes
    // Scale is at the START of each block (offset 0), weights at offset 2.
    int blocksPerRow = cols / 32;
    int blockBytes = 34;
    byte[] blocks = new byte[rows * blocksPerRow * blockBytes];
    rnd.nextBytes(blocks);
    ByteBuffer buffer = ByteBuffer.wrap(blocks).order(ByteOrder.LITTLE_ENDIAN);
    short scale = Float.floatToFloat16(0.01f);
    for (int b = 0; b < rows * blocksPerRow; b++) {
      int offset = b * blockBytes;
      buffer.putShort(offset, scale);
    }
    return MemorySegment.ofArray(blocks);
  }
}
