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
package com.integrallis.vectors.bench.q8;

import com.integrallis.vectors.core.MathUtil;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.stream.IntStream;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

/** Benchmark-only experiment that removes GGUF's scale gaps from immutable Q8_0 weights. */
final class PreparedQ8Weights {

  private static final int BLOCK_SIZE = 32;
  private static final int INTERLEAVED_BLOCK_BYTES = 34;
  private static final ValueLayout.OfShort GGUF_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final int VECTOR_BITS = IntVector.SPECIES_PREFERRED.vectorBitSize();

  private final int rows;
  private final int cols;
  private final int blocks;
  private final byte[] quants;
  private final float[] scales;

  private PreparedQ8Weights(int rows, int cols, int blocks, byte[] quants, float[] scales) {
    this.rows = rows;
    this.cols = cols;
    this.blocks = blocks;
    this.quants = quants;
    this.scales = scales;
  }

  static PreparedQ8Weights from(MemorySegment interleavedWeights, int rows, int cols) {
    Objects.requireNonNull(interleavedWeights, "interleavedWeights");
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive: " + rows);
    }
    if (cols < BLOCK_SIZE || cols % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "cols must be a positive multiple of " + BLOCK_SIZE + ": " + cols);
    }

    int blocks = cols / BLOCK_SIZE;
    int totalBlocks = Math.multiplyExact(rows, blocks);
    long requiredBytes = Math.multiplyExact((long) totalBlocks, INTERLEAVED_BLOCK_BYTES);
    if (interleavedWeights.byteSize() < requiredBytes) {
      throw new IllegalArgumentException(
          "interleavedWeights has "
              + interleavedWeights.byteSize()
              + " bytes; expected at least "
              + requiredBytes);
    }

    byte[] quants = new byte[Math.multiplyExact(totalBlocks, BLOCK_SIZE)];
    float[] scales = new float[totalBlocks];
    MemorySegment quantSegment = MemorySegment.ofArray(quants);
    for (int block = 0; block < totalBlocks; block++) {
      long sourceOffset = (long) block * INTERLEAVED_BLOCK_BYTES;
      scales[block] = Float.float16ToFloat(interleavedWeights.get(GGUF_SHORT, sourceOffset));
      MemorySegment.copy(
          interleavedWeights,
          sourceOffset + Short.BYTES,
          quantSegment,
          (long) block * BLOCK_SIZE,
          BLOCK_SIZE);
    }
    return new PreparedQ8Weights(rows, cols, blocks, quants, scales);
  }

  void multiply(
      float[] queries,
      int batchSize,
      float[] out,
      byte[] activationQuants,
      float[] activationScales) {
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(activationQuants, "activationQuants");
    Objects.requireNonNull(activationScales, "activationScales");
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
    }
    int queryEntries = Math.multiplyExact(batchSize, cols);
    int outputEntries = Math.multiplyExact(batchSize, rows);
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries has " + queries.length + " entries; expected at least " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out has " + out.length + " entries; expected at least " + outputEntries);
    }
    if (activationQuants.length < queryEntries) {
      throw new IllegalArgumentException(
          "activationQuants has "
              + activationQuants.length
              + " entries; expected at least "
              + queryEntries);
    }
    int scaleEntries = Math.multiplyExact(batchSize, blocks);
    if (activationScales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "activationScales has "
              + activationScales.length
              + " entries; expected at least "
              + scaleEntries);
    }

    quantizeActivations(queries, batchSize, activationQuants, activationScales);
    IntStream.range(0, rows)
        .parallel()
        .forEach(row -> multiplyRow(row, batchSize, activationQuants, activationScales, out));
  }

  private void multiplyRow(
      int row, int batchSize, byte[] activationQuants, float[] activationScales, float[] out) {
    int rowBlockOffset = row * blocks;
    int rowQuantOffset = row * cols;
    for (int batch = 0; batch < batchSize; batch++) {
      out[batch * rows + row] = 0.0f;
    }
    for (int block = 0; block < blocks; block++) {
      accumulateBlock(
          row,
          block,
          batchSize,
          quants,
          rowQuantOffset + block * BLOCK_SIZE,
          scales[rowBlockOffset + block],
          activationQuants,
          activationScales,
          out);
    }
  }

  private void quantizeActivations(
      float[] queries, int batchSize, byte[] activationQuants, float[] activationScales) {
    for (int batch = 0; batch < batchSize; batch++) {
      int batchQuantOffset = batch * cols;
      int batchScaleOffset = batch * blocks;
      for (int block = 0; block < blocks; block++) {
        int quantOffset = batchQuantOffset + block * BLOCK_SIZE;
        float absoluteMax = 0.0f;
        for (int lane = 0; lane < BLOCK_SIZE; lane++) {
          absoluteMax = Math.max(absoluteMax, Math.abs(queries[quantOffset + lane]));
        }
        float scale = absoluteMax / 127.0f;
        float inverseScale = absoluteMax == 0.0f ? 0.0f : 127.0f / absoluteMax;
        activationScales[batchScaleOffset + block] =
            Float.float16ToFloat(Float.floatToFloat16(scale));
        for (int lane = 0; lane < BLOCK_SIZE; lane++) {
          activationQuants[quantOffset + lane] =
              (byte) ggmlNearestInt(queries[quantOffset + lane] * inverseScale);
        }
      }
    }
  }

  private void accumulateBlock(
      int row,
      int block,
      int batchSize,
      byte[] weightQuants,
      int weightOffset,
      float weightScale,
      byte[] activationQuants,
      float[] activationScales,
      float[] out) {
    if (VECTOR_BITS >= 512) {
      IntVector weight0 =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_128, weightQuants, weightOffset)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
      IntVector weight1 =
          (IntVector)
              ByteVector.fromArray(ByteVector.SPECIES_128, weightQuants, weightOffset + 16)
                  .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
      for (int batch = 0; batch < batchSize; batch++) {
        int activationOffset = batch * cols + block * BLOCK_SIZE;
        IntVector activation0 =
            (IntVector)
                ByteVector.fromArray(ByteVector.SPECIES_128, activationQuants, activationOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector activation1 =
            (IntVector)
                ByteVector.fromArray(
                        ByteVector.SPECIES_128, activationQuants, activationOffset + 16)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector accumulator = weight0.mul(activation0).add(weight1.mul(activation1));
        accumulateResult(
            row,
            block,
            batch,
            weightScale,
            activationScales,
            accumulator.reduceLanes(VectorOperators.ADD),
            out);
      }
      return;
    }
    if (VECTOR_BITS >= 256) {
      IntVector weight0 = load8(weightQuants, weightOffset);
      IntVector weight1 = load8(weightQuants, weightOffset + 8);
      IntVector weight2 = load8(weightQuants, weightOffset + 16);
      IntVector weight3 = load8(weightQuants, weightOffset + 24);
      for (int batch = 0; batch < batchSize; batch++) {
        int activationOffset = batch * cols + block * BLOCK_SIZE;
        IntVector accumulator = weight0.mul(load8(activationQuants, activationOffset));
        accumulator = accumulator.add(weight1.mul(load8(activationQuants, activationOffset + 8)));
        accumulator = accumulator.add(weight2.mul(load8(activationQuants, activationOffset + 16)));
        accumulator = accumulator.add(weight3.mul(load8(activationQuants, activationOffset + 24)));
        accumulateResult(
            row,
            block,
            batch,
            weightScale,
            activationScales,
            accumulator.reduceLanes(VectorOperators.ADD),
            out);
      }
      return;
    }

    for (int batch = 0; batch < batchSize; batch++) {
      int activationOffset = batch * cols + block * BLOCK_SIZE;
      int integerSum = 0;
      for (int lane = 0; lane < BLOCK_SIZE; lane++) {
        integerSum += weightQuants[weightOffset + lane] * activationQuants[activationOffset + lane];
      }
      accumulateResult(row, block, batch, weightScale, activationScales, integerSum, out);
    }
  }

  private static IntVector load8(byte[] values, int offset) {
    return (IntVector)
        ByteVector.fromArray(ByteVector.SPECIES_64, values, offset)
            .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
  }

  private void accumulateResult(
      int row,
      int block,
      int batch,
      float weightScale,
      float[] activationScales,
      int integerSum,
      float[] out) {
    int outputIndex = batch * rows + row;
    float scale = weightScale * activationScales[batch * blocks + block];
    out[outputIndex] = MathUtil.fma(scale, integerSum, out[outputIndex]);
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }

  int quantByteCount() {
    return quants.length;
  }

  int scaleCount() {
    return scales.length;
  }
}
