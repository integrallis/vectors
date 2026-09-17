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

import com.integrallis.vectors.core.GgufBandGemmTest.Format;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

/**
 * Fresh-JVM probe for {@link GgufBatchDispatchTest}: with {@code
 * -Dvectors.gguf.batchedMatmulKernel=dispatch} the property-default entry points (including the
 * Q6_K tile overload Models calls) must route by batch, and the exit report must count exactly the
 * default calls made after the reset.
 */
final class GgufBatchDispatchProbe {

  private GgufBatchDispatchProbe() {}

  public static void main(String[] args) {
    System.out.println(
        "ggufBatchedMatmulKernel=" + VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel());
    check(
        GgufBatchedMatmulKernel.active() == GgufBatchedMatmulKernel.BATCH_DISPATCH,
        "active() != dispatch");
    record Call(Format format, int batch, boolean tileOverload) {}
    Call[] calls = {
      new Call(Format.Q4_K, 3, false),
      new Call(Format.Q4_K, 4, false),
      new Call(Format.Q6_K, 31, false),
      new Call(Format.Q6_K, 3, true),
      new Call(Format.Q6_K, 32, true),
      new Call(Format.Q8_0, 3, false),
      new Call(Format.Q8_0, 4, false),
    };
    int rows = 5;
    MemorySegment[] weights = new MemorySegment[calls.length];
    float[][] queries = new float[calls.length][];
    float[][] expected = new float[calls.length][];
    for (int i = 0; i < calls.length; i++) {
      Call call = calls[i];
      int cols = cols(call.format());
      Random random = new Random(i + 1L);
      weights[i] =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(call.format(), random, rows, cols));
      queries[i] = GgufBatchDispatchTest.gaussian(random, call.batch() * cols);
      boolean band =
          call.batch() >= GgufBatchedMatmulKernel.dispatchMinBatch(call.format().ordinal());
      expected[i] =
          GgufBatchDispatchTest.single(
              call.format(),
              band ? GgufBatchedMatmulKernel.BAND_F32 : GgufBatchedMatmulKernel.INTEGER,
              weights[i],
              queries[i],
              call.batch(),
              rows,
              cols);
    }
    GgufBatchedMatmulKernel.resetRoutingCounts();
    for (int i = 0; i < calls.length; i++) {
      Call call = calls[i];
      int cols = cols(call.format());
      int batch = call.batch();
      float[] out = new float[batch * rows];
      byte[] quants = new byte[batch * cols];
      float[] scales = new float[batch * (cols / call.format().blockSize)];
      switch (call.format()) {
        case Q4_K ->
            VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
                queries[i],
                weights[i],
                batch,
                rows,
                cols,
                out,
                quants,
                scales,
                new short[batch * (cols / 16)]);
        case Q6_K -> {
          if (call.tileOverload()) {
            VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
                queries[i],
                weights[i],
                batch,
                rows,
                cols,
                out,
                quants,
                scales,
                GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
          } else {
            VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
                queries[i], weights[i], batch, rows, cols, out, quants, scales);
          }
        }
        case Q8_0 ->
            VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
                queries[i], weights[i], batch, rows, cols, out, quants, scales);
      }
      check(Arrays.equals(out, expected[i]), call + " routed to the wrong arm");
      System.out.println(call + " ok");
    }
    // The exit report is printed by the shutdown hook registered for REPORT_PROPERTY.
  }

  private static int cols(Format format) {
    return format == Format.Q8_0 ? 96 : 256;
  }

  private static void check(boolean condition, String message) {
    if (!condition) {
      System.out.println("FAIL " + message);
      System.exit(1);
    }
  }
}
