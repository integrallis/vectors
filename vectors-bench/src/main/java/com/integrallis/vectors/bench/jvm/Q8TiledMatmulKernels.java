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
package com.integrallis.vectors.bench.jvm;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

/** Benchmark-only two-dimensional tiling experiments for Q8_0 by Q8_0 matrix multiplication. */
public final class Q8TiledMatmulKernels {

  private static final int BLOCK_SIZE = 32;
  private static final int BLOCK_BYTES = 34;
  private static final ValueLayout.OfShort LITTLE_ENDIAN_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private Q8TiledMatmulKernels() {}

  /** Reusable tile-local state, deliberately kept outside the timed kernel's allocation path. */
  public static final class Workspace {
    private final int rowCapacity;
    private final int batchCapacity;
    private final float[] sums;
    private final float[] weightScales;
    private final IntVector[] weight0;
    private final IntVector[] weight1;
    private final IntVector[] weight2;
    private final IntVector[] weight3;

    private Workspace(int rowCapacity, int batchCapacity) {
      if (rowCapacity < 1 || batchCapacity < 1) {
        throw new IllegalArgumentException("tile capacities must be positive");
      }
      this.rowCapacity = rowCapacity;
      this.batchCapacity = batchCapacity;
      this.sums = new float[Math.multiplyExact(rowCapacity, batchCapacity)];
      this.weightScales = new float[rowCapacity];
      this.weight0 = new IntVector[rowCapacity];
      this.weight1 = new IntVector[rowCapacity];
      this.weight2 = new IntVector[rowCapacity];
      this.weight3 = new IntVector[rowCapacity];
    }
  }

  /**
   * Allocates state reusable across calls for every tile no larger than the supplied capacities.
   */
  public static Workspace workspace(int rowCapacity, int batchCapacity) {
    return new Workspace(rowCapacity, batchCapacity);
  }

  /**
   * Multiplies row-major Q8_0 weights by a batch of prequantized row-major Q8_0 activations.
   *
   * <p>Each K block loads every weight row and activation row once per two-dimensional output tile.
   * Partial outputs remain in tile-local storage across K and are written to {@code out} once.
   */
  public static void tiled(
      MemorySegment weights,
      int batchSize,
      int rows,
      int cols,
      byte[] activationQuants,
      float[] activationScales,
      float[] out,
      int rowTile,
      int batchTile,
      Workspace workspace) {
    if (cols < BLOCK_SIZE || cols % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException("cols must be a positive multiple of 32: " + cols);
    }
    if (rowTile < 1 || rowTile > workspace.rowCapacity) {
      throw new IllegalArgumentException("rowTile exceeds workspace capacity: " + rowTile);
    }
    if (batchTile < 1 || batchTile > workspace.batchCapacity) {
      throw new IllegalArgumentException("batchTile exceeds workspace capacity: " + batchTile);
    }

    int blocks = cols / BLOCK_SIZE;
    long rowBytes = (long) blocks * BLOCK_BYTES;
    for (int rowBase = 0; rowBase < rows; rowBase += rowTile) {
      int activeRows = Math.min(rowTile, rows - rowBase);
      for (int batchBase = 0; batchBase < batchSize; batchBase += batchTile) {
        int activeBatches = Math.min(batchTile, batchSize - batchBase);
        Arrays.fill(workspace.sums, 0, activeRows * batchTile, 0.0f);

        for (int block = 0; block < blocks; block++) {
          loadWeightTile(weights, rowBase, activeRows, rowBytes, block, workspace);

          for (int batchIndex = 0; batchIndex < activeBatches; batchIndex++) {
            int batch = batchBase + batchIndex;
            int quantOffset = batch * cols + block * BLOCK_SIZE;
            IntVector activation0 = loadActivation(activationQuants, quantOffset);
            IntVector activation1 = loadActivation(activationQuants, quantOffset + 8);
            IntVector activation2 = loadActivation(activationQuants, quantOffset + 16);
            IntVector activation3 = loadActivation(activationQuants, quantOffset + 24);
            float activationScale = activationScales[batch * blocks + block];

            for (int rowIndex = 0; rowIndex < activeRows; rowIndex++) {
              IntVector integerLanes =
                  workspace
                      .weight0[rowIndex]
                      .mul(activation0)
                      .add(workspace.weight1[rowIndex].mul(activation1))
                      .add(workspace.weight2[rowIndex].mul(activation2))
                      .add(workspace.weight3[rowIndex].mul(activation3));
              int integerSum = integerLanes.reduceLanes(VectorOperators.ADD);
              int sumIndex = rowIndex * batchTile + batchIndex;
              workspace.sums[sumIndex] =
                  Math.fma(
                      workspace.weightScales[rowIndex] * activationScale,
                      integerSum,
                      workspace.sums[sumIndex]);
            }
          }
        }

        for (int rowIndex = 0; rowIndex < activeRows; rowIndex++) {
          int row = rowBase + rowIndex;
          for (int batchIndex = 0; batchIndex < activeBatches; batchIndex++) {
            int batch = batchBase + batchIndex;
            out[batch * rows + row] = workspace.sums[rowIndex * batchTile + batchIndex];
          }
        }
      }
    }
  }

  private static void loadWeightTile(
      MemorySegment weights,
      int rowBase,
      int activeRows,
      long rowBytes,
      int block,
      Workspace workspace) {
    for (int rowIndex = 0; rowIndex < activeRows; rowIndex++) {
      long blockOffset = (rowBase + rowIndex) * rowBytes + (long) block * BLOCK_BYTES;
      workspace.weightScales[rowIndex] =
          Float.float16ToFloat(weights.get(LITTLE_ENDIAN_SHORT, blockOffset));
      long quantOffset = blockOffset + Short.BYTES;
      workspace.weight0[rowIndex] = loadWeight(weights, quantOffset);
      workspace.weight1[rowIndex] = loadWeight(weights, quantOffset + 8);
      workspace.weight2[rowIndex] = loadWeight(weights, quantOffset + 16);
      workspace.weight3[rowIndex] = loadWeight(weights, quantOffset + 24);
    }
  }

  private static IntVector loadWeight(MemorySegment weights, long offset) {
    return (IntVector)
        ByteVector.fromMemorySegment(
                ByteVector.SPECIES_64, weights, offset, ByteOrder.LITTLE_ENDIAN)
            .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
  }

  private static IntVector loadActivation(byte[] quants, int offset) {
    return (IntVector)
        ByteVector.fromArray(ByteVector.SPECIES_64, quants, offset)
            .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
  }
}
