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

import java.util.Objects;

/** Caller-owned reusable Q8_0 activation storage for staged GGUF matrix operations. */
public final class GgufQ8_0Batch {

  private static final int BLOCK_SIZE = VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
  private static final int CORRECTIONS_PER_BLOCK = BLOCK_SIZE / 4;

  private final int batchCapacity;
  private final int dimensions;
  private final int blocks;
  private final byte[] quants;
  private final float[] scales;
  private final int[] zeroPointCorrections;

  private GgufQ8_0Batch(int batchCapacity, int dimensions) {
    this.batchCapacity = batchCapacity;
    this.dimensions = dimensions;
    this.blocks = dimensions / BLOCK_SIZE;
    this.quants = new byte[checkedProduct(batchCapacity, dimensions, "Q8_0 quants")];
    this.scales = new float[checkedProduct(batchCapacity, blocks, "Q8_0 scales")];
    this.zeroPointCorrections =
        new int
            [checkedProduct(
                checkedProduct(batchCapacity, blocks, "Q8_0 correction blocks"),
                CORRECTIONS_PER_BLOCK,
                "Q8_0 corrections")];
  }

  /** Allocates reusable storage for {@code batchCapacity} activation rows. */
  public static GgufQ8_0Batch allocate(int batchCapacity, int dimensions) {
    if (batchCapacity < 1) {
      throw new IllegalArgumentException("batchCapacity must be positive: " + batchCapacity);
    }
    if (dimensions < BLOCK_SIZE || dimensions % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "dimensions must be a positive multiple of " + BLOCK_SIZE + ": " + dimensions);
    }
    return new GgufQ8_0Batch(batchCapacity, dimensions);
  }

  /** Quantizes complete batch rows for a subsequent Q4_0 operation. */
  public void quantizeForQ4(float[] values, int batchSize, GgufQ4Kernel kernel) {
    quantizeBlockRangeForQ4(values, batchSize, 0, blocks, kernel);
  }

  /**
   * Quantizes one non-empty contiguous block range across active batch rows.
   *
   * <p>Distinct block ranges may be written concurrently.
   */
  public void quantizeBlockRangeForQ4(
      float[] values, int batchSize, int fromBlock, int toBlock, GgufQ4Kernel kernel) {
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(kernel, "kernel");
    checkBatchSize(batchSize);
    if (fromBlock < 0 || fromBlock >= toBlock || toBlock > blocks) {
      throw new IndexOutOfBoundsException(
          "block range must satisfy 0 <= fromBlock < toBlock <= blocks: "
              + fromBlock
              + ".."
              + toBlock
              + " for "
              + blocks);
    }
    checkValuesLength(values, batchSize);
    quantizeForQ4(values, 0, batchSize, fromBlock, toBlock, kernel);
  }

  /**
   * Quantizes one non-empty contiguous range of active batch rows.
   *
   * <p>Distinct batch ranges may be written concurrently.
   */
  public void quantizeBatchRangeForQ4(
      float[] values, int batchSize, int fromBatch, int toBatch, GgufQ4Kernel kernel) {
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(kernel, "kernel");
    checkBatchSize(batchSize);
    if (fromBatch < 0 || fromBatch >= toBatch || toBatch > batchSize) {
      throw new IndexOutOfBoundsException(
          "batch range must satisfy 0 <= fromBatch < toBatch <= batchSize: "
              + fromBatch
              + ".."
              + toBatch
              + " for "
              + batchSize);
    }
    checkValuesLength(values, batchSize);
    quantizeForQ4(values, fromBatch, toBatch, 0, blocks, kernel);
  }

  private void quantizeForQ4(
      float[] values, int fromBatch, int toBatch, int fromBlock, int toBlock, GgufQ4Kernel kernel) {
    int length = (toBlock - fromBlock) * BLOCK_SIZE;
    boolean corrections = kernel == GgufQ4Kernel.UNSIGNED_PAIRWISE;
    for (int batch = fromBatch; batch < toBatch; batch++) {
      int valueOffset = batch * dimensions + fromBlock * BLOCK_SIZE;
      int scaleOffset = batch * blocks + fromBlock;
      if (corrections) {
        GgufQuantizationSupport.quantizeQ8_0WithQ4Corrections(
            values,
            valueOffset,
            length,
            quants,
            valueOffset,
            scales,
            scaleOffset,
            zeroPointCorrections,
            scaleOffset * CORRECTIONS_PER_BLOCK);
      } else {
        GgufQuantizationSupport.quantizeQ8_0(
            values, valueOffset, length, quants, valueOffset, scales, scaleOffset);
      }
    }
  }

  private void checkValuesLength(float[] values, int batchSize) {
    int requiredValues = checkedProduct(batchSize, dimensions, "batchSize * dimensions");
    if (values.length < requiredValues) {
      throw new IllegalArgumentException(
          "values.length must be >= batchSize * dimensions: "
              + values.length
              + " < "
              + requiredValues);
    }
  }

  /** Returns the maximum number of activation rows retained by this storage. */
  public int batchCapacity() {
    return batchCapacity;
  }

  /** Returns the number of values in each activation row. */
  public int dimensions() {
    return dimensions;
  }

  byte[] quants() {
    return quants;
  }

  float[] scales() {
    return scales;
  }

  int[] zeroPointCorrections() {
    return zeroPointCorrections;
  }

  private void checkBatchSize(int batchSize) {
    if (batchSize < 1 || batchSize > batchCapacity) {
      throw new IllegalArgumentException(
          "batchSize must be between 1 and " + batchCapacity + ": " + batchSize);
    }
  }

  private static int checkedProduct(int first, int second, String label) {
    try {
      return Math.multiplyExact(first, second);
    } catch (ArithmeticException exception) {
      throw new IllegalArgumentException(label + " exceeds array capacity", exception);
    }
  }
}
