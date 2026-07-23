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
  private static final int CORRECTIONS_PER_BLOCK = BLOCK_SIZE / 8;

  private final int batchCapacity;
  private final int dimensions;
  private final int blocks;
  private final GgufQ8ActivationLayout layout;
  private final byte[] quants;
  private final byte[] blockMajorQuants;
  private final float[] scales;
  private final int[] zeroPointCorrections;

  private GgufQ8_0Batch(int batchCapacity, int dimensions, GgufQ8ActivationLayout layout) {
    this.batchCapacity = batchCapacity;
    this.dimensions = dimensions;
    this.blocks = dimensions / BLOCK_SIZE;
    this.layout = layout;
    this.quants = new byte[checkedProduct(batchCapacity, dimensions, "Q8_0 quants")];
    this.blockMajorQuants =
        layout == GgufQ8ActivationLayout.BLOCK_MAJOR_BYTES
            ? new byte[checkedProduct(batchCapacity, dimensions, "block-major Q8_0 quants")]
            : null;
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
    return allocate(batchCapacity, dimensions, GgufQ8ActivationLayout.PACKED_BYTES);
  }

  /** Allocates reusable storage with an explicit retained activation layout. */
  public static GgufQ8_0Batch allocate(
      int batchCapacity, int dimensions, GgufQ8ActivationLayout layout) {
    if (batchCapacity < 1) {
      throw new IllegalArgumentException("batchCapacity must be positive: " + batchCapacity);
    }
    if (dimensions < BLOCK_SIZE || dimensions % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "dimensions must be a positive multiple of " + BLOCK_SIZE + ": " + dimensions);
    }
    return new GgufQ8_0Batch(batchCapacity, dimensions, Objects.requireNonNull(layout, "layout"));
  }

  /** Quantizes complete batch rows for a subsequent Q4_0 operation. */
  public void quantizeForQ4(float[] values, int batchSize, GgufQ4Kernel kernel) {
    quantizeBlockRangeForQ4(values, batchSize, 0, blocks, kernel);
  }

  /** Quantizes complete batch rows for a subsequent Q8_0 operation. */
  public void quantize(float[] values, int batchSize) {
    quantizeBlockRange(values, batchSize, 0, blocks);
  }

  /**
   * Quantizes one non-empty contiguous range of active batch rows for Q8_0 consumers.
   *
   * <p>Distinct batch ranges may be written concurrently.
   */
  public void quantizeBatchRange(float[] values, int batchSize, int fromBatch, int toBatch) {
    quantizeRange(values, batchSize, fromBatch, toBatch, 0, blocks, false);
  }

  /**
   * Quantizes one non-empty contiguous block range across active batch rows.
   *
   * <p>Distinct block ranges may be written concurrently.
   */
  public void quantizeBlockRange(float[] values, int batchSize, int fromBlock, int toBlock) {
    quantizeBlockRange(values, batchSize, fromBlock, toBlock, false);
  }

  /**
   * Quantizes one non-empty contiguous block range across active batch rows.
   *
   * <p>Distinct block ranges may be written concurrently.
   */
  public void quantizeBlockRangeForQ4(
      float[] values, int batchSize, int fromBlock, int toBlock, GgufQ4Kernel kernel) {
    Objects.requireNonNull(kernel, "kernel");
    quantizeBlockRange(
        values, batchSize, fromBlock, toBlock, kernel == GgufQ4Kernel.UNSIGNED_PAIRWISE);
  }

  private void quantizeBlockRange(
      float[] values, int batchSize, int fromBlock, int toBlock, boolean corrections) {
    quantizeRange(values, batchSize, 0, batchSize, fromBlock, toBlock, corrections);
  }

  private void quantizeRange(
      float[] values,
      int batchSize,
      int fromBatch,
      int toBatch,
      int fromBlock,
      int toBlock,
      boolean corrections) {
    Objects.requireNonNull(values, "values");
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
    if (fromBlock < 0 || fromBlock >= toBlock || toBlock > blocks) {
      throw new IndexOutOfBoundsException(
          "block range must satisfy 0 <= fromBlock < toBlock <= blocks: "
              + fromBlock
              + ".."
              + toBlock
              + " for "
              + blocks);
    }
    int length = (toBlock - fromBlock) * BLOCK_SIZE;
    int requiredValues = checkedProduct(batchSize, dimensions, "batchSize * dimensions");
    if (values.length < requiredValues) {
      throw new IllegalArgumentException(
          "values.length must be >= batchSize * dimensions: "
              + values.length
              + " < "
              + requiredValues);
    }
    for (int batch = fromBatch; batch < toBatch; batch++) {
      int valueOffset = batch * dimensions + fromBlock * BLOCK_SIZE;
      int scaleOffset = batch * blocks + fromBlock;
      if (corrections) {
        GgufQuantizationSupport.quantizeQ8_0WithCombinedQ4Corrections(
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
      if (blockMajorQuants != null) {
        for (int block = fromBlock; block < toBlock; block++) {
          int sourceOffset = batch * dimensions + block * BLOCK_SIZE;
          int blockMajorOffset = (block * batchCapacity + batch) * BLOCK_SIZE;
          System.arraycopy(quants, sourceOffset, blockMajorQuants, blockMajorOffset, BLOCK_SIZE);
        }
      }
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

  /** Returns the retained activation layout. */
  public GgufQ8ActivationLayout layout() {
    return layout;
  }

  byte[] quants() {
    return quants;
  }

  byte[] blockMajorQuants() {
    return blockMajorQuants;
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
