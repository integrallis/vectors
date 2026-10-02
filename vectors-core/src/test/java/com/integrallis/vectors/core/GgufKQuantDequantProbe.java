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
 * Fresh-JVM probe for {@link GgufKQuantDequantTest}: with {@code -Dvectors.maxBits} and {@code
 * -Dvectors.gguf.band.dequant} set, (1) the configuration reports the requested and effective arm,
 * (2) the band arm's default path runs the effective arm and only it (call counters), (3) the band
 * matmul output is bit-identical to the band matmul with the scalar dequant, and (4) every arm is
 * bit-identical to scalar on the adversarial and random blocks at this species width.
 */
final class GgufKQuantDequantProbe {

  private GgufKQuantDequantProbe() {}

  public static void main(String[] args) {
    System.out.println("bandDequant=" + GgufBatchedMatmulKernel.bandDequantConfiguration());
    System.out.println("band=" + GgufBatchedMatmulKernel.bandConfiguration());
    GgufKQuantDequant.Arm effective = GgufKQuantDequant.ACTIVE;
    check(GgufKQuantDequant.COUNTING, "counting disabled");
    int[][] shapes = {{1, 7, 768}, {4, 97, 2560}, {5, 512, 2560}, {2, 3, 8192}};
    int[] formats = {GgufBandGemm.Q4_K, GgufBandGemm.Q6_K};
    float[][][] viaDefault = new float[formats.length][shapes.length][];
    // Default path first, so the counters see only the configured arm.
    for (int f = 0; f < formats.length; f++) {
      for (int s = 0; s < shapes.length; s++) {
        viaDefault[f][s] = runBand(formats[f], shapes[s], null);
      }
    }
    for (GgufKQuantDequant.Arm arm : GgufKQuantDequant.Arm.values()) {
      long calls = GgufKQuantDequant.calls(arm);
      System.out.println("calls " + arm + "=" + calls);
      check(arm == effective ? calls > 0 : calls == 0, "calls for " + arm + " = " + calls);
    }
    for (int f = 0; f < formats.length; f++) {
      for (int s = 0; s < shapes.length; s++) {
        float[] scalar = runBand(formats[f], shapes[s], GgufKQuantDequant.Arm.SCALAR);
        check(
            Arrays.equals(
                GgufKQuantDequantTest.bits(viaDefault[f][s]), GgufKQuantDequantTest.bits(scalar)),
            "band output differs from scalar-dequant band: format "
                + formats[f]
                + " shape "
                + Arrays.toString(shapes[s]));
      }
    }
    // Repeated passes so the comparison covers C2-compiled kernels, not only the interpreter.
    for (int pass = 0; pass < 40; pass++) {
      for (GgufKQuantDequant.Arm arm : GgufKQuantDequant.Arm.values()) {
        for (int format : formats) {
          String mismatch =
              GgufKQuantDequantTest.firstMismatch(
                  arm,
                  format,
                  GgufKQuantDequantTest.testBlocks(format, new Random(5L + format + pass)));
          check(
              mismatch == null, "pass " + pass + " " + arm + " format " + format + ": " + mismatch);
        }
      }
    }
    System.out.println("probe ok");
  }

  private static float[] runBand(int format, int[] shape, GgufKQuantDequant.Arm arm) {
    int batch = shape[0];
    int rows = shape[1];
    int cols = shape[2];
    GgufBandGemmTest.Format testFormat =
        format == GgufBandGemm.Q4_K ? GgufBandGemmTest.Format.Q4_K : GgufBandGemmTest.Format.Q6_K;
    Random random = new Random(31L * format + rows + cols);
    MemorySegment weights =
        MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(testFormat, random, rows, cols));
    float[] queries = new float[batch * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = (float) random.nextGaussian();
    }
    if (arm == null) {
      return GgufBandGemmTest.run(
          testFormat,
          GgufBandGemmTest.VectorUtilSupportKind.BAND,
          weights,
          queries,
          batch,
          rows,
          cols);
    }
    float[] out = new float[batch * rows];
    GgufBandGemm.gemm(format, arm, queries, weights, batch, rows, cols, out);
    return out;
  }

  private static void check(boolean condition, String message) {
    if (!condition) {
      System.out.println("FAIL " + message);
      System.exit(1);
    }
  }
}
