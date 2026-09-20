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

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The scalar and Vector API paths for Q6_K must produce bit-identical results, and until now they
 * did not.
 *
 * <p>The scalar path kept eight float lane accumulators, documented as existing so its reduction
 * order would match the eight-lane order of the Vector API path. That was true of an older SIMD
 * implementation. The current one reduces each super-block to a single integer and applies one
 * fused multiply per block, so the two were folding in genuinely different orders: eight roundings
 * per block against one. They disagreed by one to two units in the last place at every width.
 *
 * <p>Nothing caught it because the existing scalar-versus-Panama comparison for Q6_K carries a
 * tolerance of 1e-3, which such a disagreement passes without trace. These tests assert bit
 * equality instead, and include Q4_K as a control so a failure here cannot be blamed on the harness
 * or on the Vector API in general.
 */
@Tag("unit")
class Q6KReductionOrderContractTest {

  private static final int BLOCK = 256;
  private static final int BLOCK_BYTES = 210;
  private static final int Q4_K_BLOCK_BYTES = 144;

  @Test
  @DisplayName("one super-block: scalar and Panama agree bit for bit")
  void oneSuperBlockIsBitIdentical() {
    assertBitIdentical(1, 4);
  }

  @Test
  @DisplayName("two super-blocks: the cross-block fold must not change a single bit")
  void twoSuperBlocksAreBitIdentical() {
    assertBitIdentical(2, 4);
  }

  @Test
  @DisplayName("eight super-blocks: a realistic projection width stays bit-identical")
  void eightSuperBlocksAreBitIdentical() {
    assertBitIdentical(8, 4);
  }

  @Test
  @DisplayName("control: Q4_K is bit-identical at the same widths, so the harness is sound")
  void q4kIsBitIdenticalAtEveryWidth() {
    for (int blocks : new int[] {1, 2, 8}) {
      assertQ4KBitIdentical(blocks, 4);
    }
  }

  /**
   * The control that makes the Q6_K result mean something. Q4_K's contract is two fused multiplies
   * per super-block in ascending order, and the device bisect found it exact from one super-block
   * to forty-eight. If Q4_K is bit-identical here and Q6_K is not, the divergence belongs to Q6_K
   * rather than to this harness or to the Vector API generally.
   */
  private static void assertQ4KBitIdentical(int blocksPerRow, int rows) {
    int cols = blocksPerRow * BLOCK;
    Random random = new Random(20260920L + blocksPerRow);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = (random.nextFloat() - 0.5f) * 2.0f;
    }
    byte[] weights = new byte[rows * blocksPerRow * Q4_K_BLOCK_BYTES];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += Q4_K_BLOCK_BYTES) {
      buffer.putShort(offset, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
      buffer.putShort(offset + 2, Float.floatToFloat16(random.nextFloat() * 0.01f));
    }
    MemorySegment weightSegment = MemorySegment.ofArray(weights);
    float[] scalarOut = new float[rows];
    float[] panamaOut = new float[rows];
    byte[] scalarQuants = new byte[cols];
    byte[] panamaQuants = new byte[cols];
    float[] scalarScales = new float[blocksPerRow];
    float[] panamaScales = new float[blocksPerRow];
    short[] scalarSums = new short[cols / 16];
    short[] panamaSums = new short[cols / 16];

    new ScalarVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, scalarOut, scalarQuants, scalarScales, scalarSums);
    new PanamaVectorUtilSupport()
        .ggufQ4_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, panamaOut, panamaQuants, panamaScales, panamaSums);

    for (int row = 0; row < rows; row++) {
      assertThat(Float.floatToRawIntBits(panamaOut[row]))
          .describedAs(
              "Q4_K row %d at %d super-block(s): scalar %s vs panama %s",
              row, blocksPerRow, scalarOut[row], panamaOut[row])
          .isEqualTo(Float.floatToRawIntBits(scalarOut[row]));
    }
  }

  private static void assertBitIdentical(int blocksPerRow, int rows) {
    int cols = blocksPerRow * BLOCK;
    Random random = new Random(20260920L + blocksPerRow);

    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = (random.nextFloat() - 0.5f) * 2.0f;
    }

    byte[] weights = new byte[rows * blocksPerRow * BLOCK_BYTES];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += BLOCK_BYTES) {
      buffer.putShort(offset + 208, Float.floatToFloat16(random.nextFloat() * 0.05f + 0.001f));
    }
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    float[] scalarOut = new float[rows];
    float[] panamaOut = new float[rows];
    byte[] scalarQuants = new byte[cols];
    byte[] panamaQuants = new byte[cols];
    float[] scalarScales = new float[blocksPerRow];
    float[] panamaScales = new float[blocksPerRow];

    new ScalarVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, scalarOut, scalarQuants, scalarScales);
    new PanamaVectorUtilSupport()
        .ggufQ6_KQ8_KMatVecDot(
            query, weightSegment, rows, cols, panamaOut, panamaQuants, panamaScales);

    // The activation quantisation must agree exactly before the dot products can be compared.
    assertThat(panamaQuants).containsExactly(scalarQuants);
    assertThat(panamaScales).containsExactly(scalarScales);

    for (int row = 0; row < rows; row++) {
      assertThat(Float.floatToRawIntBits(panamaOut[row]))
          .describedAs(
              "row %d at %d super-block(s): scalar %s (bits %d) vs panama %s (bits %d), ulps %d",
              row,
              blocksPerRow,
              scalarOut[row],
              Float.floatToRawIntBits(scalarOut[row]),
              panamaOut[row],
              Float.floatToRawIntBits(panamaOut[row]),
              Math.abs(
                  (long) Float.floatToRawIntBits(panamaOut[row])
                      - Float.floatToRawIntBits(scalarOut[row])))
          .isEqualTo(Float.floatToRawIntBits(scalarOut[row]));
    }
  }
}
