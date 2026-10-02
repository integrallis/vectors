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

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

/**
 * Fresh-JVM probe for {@link GgufBandGemmTest}: with {@code
 * -Dvectors.gguf.batchedMatmulKernel=band} the default entry points must run exactly the band arm,
 * for all three formats, and the capabilities must say so.
 */
final class GgufBandGemmSwitchProbe {

  private GgufBandGemmSwitchProbe() {}

  public static void main(String[] args) {
    String reported = VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel();
    System.out.println("ggufBatchedMatmulKernel=" + reported);
    check(GgufBatchedMatmulKernel.active() == GgufBatchedMatmulKernel.BAND_F32, "active() != band");
    for (GgufBandGemmTest.Format format : GgufBandGemmTest.Format.values()) {
      Random random = new Random(format.ordinal() + 11L);
      int batch = 5;
      int rows = 11;
      int cols = format == GgufBandGemmTest.Format.Q8_0 ? 544 : 768;
      MemorySegment weights =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(format, random, rows, cols));
      float[] queries = new float[batch * cols];
      for (int i = 0; i < queries.length; i++) {
        queries[i] = (float) random.nextGaussian();
      }
      float[] viaDefault = new float[batch * rows];
      byte[] quants = new byte[batch * cols];
      float[] scales = new float[batch * (cols / format.blockSize)];
      short[] sums = new short[batch * (cols / 16)];
      switch (format) {
        case Q4_K ->
            VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
                queries, weights, batch, rows, cols, viaDefault, quants, scales, sums);
        case Q6_K ->
            VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
                queries, weights, batch, rows, cols, viaDefault, quants, scales);
        case Q8_0 ->
            VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
                queries, weights, batch, rows, cols, viaDefault, quants, scales);
      }
      float[] band =
          GgufBandGemmTest.run(
              format,
              GgufBandGemmTest.VectorUtilSupportKind.BAND,
              weights,
              queries,
              batch,
              rows,
              cols);
      float[] integer =
          GgufBandGemmTest.run(
              format,
              GgufBandGemmTest.VectorUtilSupportKind.PANAMA_INTEGER,
              weights,
              queries,
              batch,
              rows,
              cols);
      check(Arrays.equals(viaDefault, band), format + ": default entry point is not the band arm");
      check(!Arrays.equals(viaDefault, integer), format + ": default entry point is integer");
      System.out.println(format + " default==band ok");
      int[][] shapes =
          format == GgufBandGemmTest.Format.Q8_0
              ? new int[][] {{1, 1, 32}, {6, 37, 544}, {3, 1600, 2560}}
              : new int[][] {{1, 5, 256}, {6, 37, 768}, {5, 512, 2560}, {2, 3, 8192}};
      for (int[] shape : shapes) {
        double use = GgufBandGemmTest.bandToleranceUse(format, shape[0], shape[1], shape[2]);
        check(use <= 1.0, format + " " + Arrays.toString(shape) + " tolerance use " + use);
        System.out.println(format + " " + Arrays.toString(shape) + " tolerance use " + use);
      }
    }
  }

  private static void check(boolean condition, String message) {
    if (!condition) {
      System.out.println("FAIL " + message);
      System.exit(1);
    }
  }
}
