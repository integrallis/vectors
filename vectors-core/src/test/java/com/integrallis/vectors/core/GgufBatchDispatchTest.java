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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.vectors.core.GgufBandGemmTest.Format;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Batch-size dispatch between the integer kernels and the band F32 arm ({@link
 * GgufBatchedMatmulKernel#BATCH_DISPATCH}).
 *
 * <p>The thresholds are the ones pre-registered and passed on 2026-09-17 (vectors-bench
 * jmh-results/2026-09-16-band-gemm, pre-registration 2): band from batch 4 for Q4_K, 32 for Q6_K
 * and 4 for Q8_0; integer below. Routing is asserted by bit-identity with the explicitly selected
 * arm at T - 1 and T, and each routed result is also held to that arm's documented tolerance
 * against the exact double reference (see {@link GgufBandGemmTest}).
 */
class GgufBatchDispatchTest {

  private static final double BAND_RELATIVE_TOLERANCE = 1e-4;
  private static final double ABSOLUTE_SLACK = 1e-6;

  @Test
  void thresholdsAreThePreRegisteredValues() {
    assertThat(GgufBatchedMatmulKernel.parse("dispatch"))
        .isEqualTo(GgufBatchedMatmulKernel.BATCH_DISPATCH);
    assertThat(GgufBatchedMatmulKernel.dispatchMinBatch(GgufBatchedMatmulKernel.FORMAT_Q4_K))
        .isEqualTo(4);
    assertThat(GgufBatchedMatmulKernel.dispatchMinBatch(GgufBatchedMatmulKernel.FORMAT_Q6_K))
        .isEqualTo(32);
    assertThat(GgufBatchedMatmulKernel.dispatchMinBatch(GgufBatchedMatmulKernel.FORMAT_Q8_0))
        .isEqualTo(4);
    assertThat(GgufBatchedMatmulKernel.FORMAT_Q4_K).isEqualTo(GgufBandGemm.Q4_K);
    assertThat(GgufBatchedMatmulKernel.FORMAT_Q6_K).isEqualTo(GgufBandGemm.Q6_K);
    assertThat(GgufBatchedMatmulKernel.FORMAT_Q8_0).isEqualTo(GgufBandGemm.Q8_0);
    assertThat(GgufBatchedMatmulKernel.describeDispatch())
        .startsWith("dispatch(band-at-batch>=Q4_K:4,Q6_K:32,Q8_0:4;band=band-f32(tile=");
  }

  @Test
  void routingPredicateSwitchesExactlyAtTheThreshold() {
    for (int format :
        new int[] {
          GgufBatchedMatmulKernel.FORMAT_Q4_K,
          GgufBatchedMatmulKernel.FORMAT_Q6_K,
          GgufBatchedMatmulKernel.FORMAT_Q8_0
        }) {
      int threshold = GgufBatchedMatmulKernel.dispatchMinBatch(format);
      GgufBatchedMatmulKernel dispatch = GgufBatchedMatmulKernel.BATCH_DISPATCH;
      assertThat(dispatch.usesBand(format, 1)).isFalse();
      assertThat(dispatch.usesBand(format, threshold - 1)).isFalse();
      assertThat(dispatch.usesBand(format, threshold)).isTrue();
      assertThat(dispatch.usesBand(format, 512)).isTrue();
      assertThat(GgufBatchedMatmulKernel.INTEGER.usesBand(format, 512)).isFalse();
      assertThat(GgufBatchedMatmulKernel.BAND_F32.usesBand(format, 1)).isTrue();
    }
  }

  @ParameterizedTest
  @EnumSource(Format.class)
  void singleMatrixEntryPointRoutesByBatchAtTheBoundary(Format format) {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    int threshold = GgufBatchedMatmulKernel.dispatchMinBatch(format.ordinal());
    int rows = 7;
    int cols = format == Format.Q8_0 ? 96 : 256;
    for (int batch : new int[] {threshold - 1, threshold}) {
      Random random = new Random(1000L * format.ordinal() + batch);
      MemorySegment weights =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(format, random, rows, cols));
      float[] queries = gaussian(random, batch * cols);
      GgufBatchedMatmulKernel.resetRoutingCounts();
      float[] dispatched =
          single(
              format, GgufBatchedMatmulKernel.BATCH_DISPATCH, weights, queries, batch, rows, cols);
      long[] counts = GgufBatchedMatmulKernel.routingCounts(format.ordinal());
      float[] integer =
          single(format, GgufBatchedMatmulKernel.INTEGER, weights, queries, batch, rows, cols);
      float[] band =
          single(format, GgufBatchedMatmulKernel.BAND_F32, weights, queries, batch, rows, cols);
      boolean expectBand = batch >= threshold;

      assertThat(dispatched)
          .as("%s batch=%d routed to %s", format, batch, expectBand ? "band" : "integer")
          .containsExactly(expectBand ? band : integer);
      assertThat(Arrays.equals(band, integer)).as("arms must be distinguishable").isFalse();
      // {integer calls, integer rows, band calls, band rows} observed for the dispatched call.
      assertThat(counts)
          .containsExactly(expectBand ? new long[] {0, 0, 1, batch} : new long[] {1, batch, 0, 0});
      assertWithinArmTolerance(format, weights, queries, batch, rows, cols, dispatched, expectBand);
    }
  }

  @Test
  void groupedQ4_KEntryPointsRouteEachMatrixByItsOwnFormatThreshold() {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    int cols = 256;
    int q4Rows = 5;
    int q6Rows = 3;
    // Below Q4_K's threshold, between the thresholds (Q4_K band, Q6_K integer), and at Q6_K's.
    for (int batch : new int[] {3, 4, 31, 32}) {
      Random random = new Random(77L + batch);
      MemorySegment first =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(Format.Q4_K, random, q4Rows, cols));
      MemorySegment second =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(Format.Q4_K, random, q4Rows, cols));
      MemorySegment third =
          MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(Format.Q6_K, random, q6Rows, cols));
      float[] queries = gaussian(random, batch * cols);

      float[][] dispatched =
          triple(GgufBatchedMatmulKernel.BATCH_DISPATCH, first, second, third, queries, batch);
      float[][] integer =
          triple(GgufBatchedMatmulKernel.INTEGER, first, second, third, queries, batch);
      float[][] band =
          triple(GgufBatchedMatmulKernel.BAND_F32, first, second, third, queries, batch);
      boolean q4Band = batch >= 4;
      boolean q6Band = batch >= 32;
      assertThat(dispatched[0]).containsExactly(q4Band ? band[0] : integer[0]);
      assertThat(dispatched[1]).containsExactly(q4Band ? band[1] : integer[1]);
      assertThat(dispatched[2]).containsExactly(q6Band ? band[2] : integer[2]);
      // The mixed case runs Q6_K through the single-matrix integer kernel; it must equal the
      // grouped integer kernel's Q6_K output.
      assertThat(
              single(
                  Format.Q6_K,
                  GgufBatchedMatmulKernel.INTEGER,
                  third,
                  queries,
                  batch,
                  q6Rows,
                  cols))
          .containsExactly(integer[2]);

      float[][] dualDispatched =
          dual(GgufBatchedMatmulKernel.BATCH_DISPATCH, first, second, queries, batch);
      assertThat(dualDispatched[0]).containsExactly(q4Band ? band[0] : integer[0]);
      assertThat(dualDispatched[1]).containsExactly(q4Band ? band[1] : integer[1]);
    }
  }

  @Test
  void groupedQ8_0EntryPointsRouteByBatch() {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    int cols = 96;
    int rows = 6;
    for (int batch : new int[] {3, 4}) {
      Random random = new Random(501L + batch);
      MemorySegment[] weights = new MemorySegment[3];
      for (int i = 0; i < 3; i++) {
        weights[i] =
            MemorySegment.ofArray(GgufBandGemmTest.randomBlocks(Format.Q8_0, random, rows, cols));
      }
      float[] queries = gaussian(random, batch * cols);
      float[][] expected = new float[3][];
      for (int i = 0; i < 3; i++) {
        expected[i] =
            single(
                Format.Q8_0,
                batch >= 4 ? GgufBatchedMatmulKernel.BAND_F32 : GgufBatchedMatmulKernel.INTEGER,
                weights[i],
                queries,
                batch,
                rows,
                cols);
      }
      float[][] out = new float[3][batch * rows];
      VectorUtil.ggufQ8_0Q8_0TripleBatchedMatmul(
          queries,
          weights[0],
          rows,
          out[0],
          weights[1],
          rows,
          out[1],
          weights[2],
          rows,
          out[2],
          batch,
          cols,
          new byte[batch * cols],
          new float[batch * (cols / 32)],
          GgufBatchedMatmulKernel.BATCH_DISPATCH);
      float[][] dualOut = new float[2][batch * rows];
      VectorUtil.ggufQ8_0Q8_0DualBatchedMatmul(
          queries,
          weights[0],
          rows,
          dualOut[0],
          weights[1],
          rows,
          dualOut[1],
          batch,
          cols,
          new byte[batch * cols],
          new float[batch * (cols / 32)],
          GgufBatchedMatmulKernel.BATCH_DISPATCH);
      for (int i = 0; i < 3; i++) {
        assertThat(out[i]).as("triple %d batch %d", i, batch).containsExactly(expected[i]);
      }
      assertThat(dualOut[0]).containsExactly(expected[0]);
      assertThat(dualOut[1]).containsExactly(expected[1]);
    }
  }

  @Test
  void propertySelectsDispatchReportsThresholdsAndCountsInAFreshJvm() throws Exception {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-D" + GgufBatchedMatmulKernel.PROPERTY + "=dispatch",
                "-D" + GgufBatchedMatmulKernel.REPORT_PROPERTY + "=true",
                "-cp",
                System.getProperty("java.class.path"),
                GgufBatchDispatchProbe.class.getName())
            .redirectErrorStream(true)
            .start();
    boolean completed = process.waitFor(120, TimeUnit.SECONDS);
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    assertThat(completed).as("probe timed out; output:%n%s", output).isTrue();
    assertThat(process.exitValue()).as("probe output:%n%s", output).isZero();
    assertThat(output)
        .contains("ggufBatchedMatmulKernel=dispatch(band-at-batch>=Q4_K:4,Q6_K:32,Q8_0:4;band=")
        .contains("vectors-gguf-batched-matmul-routing mode=dispatch")
        .contains("Q4_K integer=1/3 band=1/4")
        .contains("Q6_K integer=2/34 band=1/32")
        .contains("Q8_0 integer=1/3 band=1/4");
  }

  static float[] gaussian(Random random, int size) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      values[i] = (float) random.nextGaussian();
    }
    return values;
  }

  static float[] single(
      Format format,
      GgufBatchedMatmulKernel kernel,
      MemorySegment weights,
      float[] queries,
      int batch,
      int rows,
      int cols) {
    float[] out = new float[batch * rows];
    byte[] quants = new byte[batch * cols];
    float[] scales = new float[batch * (cols / format.blockSize)];
    switch (format) {
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
              queries,
              weights,
              batch,
              rows,
              cols,
              out,
              quants,
              scales,
              new short[batch * (cols / 16)],
              kernel);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              queries, weights, batch, rows, cols, out, quants, scales, kernel);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              queries, weights, batch, rows, cols, out, quants, scales, kernel);
    }
    return out;
  }

  private static float[][] triple(
      GgufBatchedMatmulKernel kernel,
      MemorySegment first,
      MemorySegment second,
      MemorySegment third,
      float[] queries,
      int batch) {
    int cols = 256;
    float[][] out = {new float[batch * 5], new float[batch * 5], new float[batch * 3]};
    VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
        queries,
        first,
        5,
        out[0],
        second,
        5,
        out[1],
        third,
        3,
        out[2],
        batch,
        cols,
        new byte[batch * cols],
        new float[batch],
        new short[batch * 16],
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
        kernel);
    return out;
  }

  private static float[][] dual(
      GgufBatchedMatmulKernel kernel,
      MemorySegment first,
      MemorySegment second,
      float[] queries,
      int batch) {
    int cols = 256;
    float[][] out = {new float[batch * 5], new float[batch * 5]};
    VectorUtil.ggufQ4_KQ8_KDualBatchedMatmul(
        queries,
        first,
        5,
        out[0],
        second,
        5,
        out[1],
        batch,
        cols,
        new byte[batch * cols],
        new float[batch],
        new short[batch * 16],
        kernel);
    return out;
  }

  private static void assertWithinArmTolerance(
      Format format,
      MemorySegment weights,
      float[] queries,
      int batch,
      int rows,
      int cols,
      float[] out,
      boolean band) {
    double[] w = GgufBandGemmTest.dequantizeReference(format, weights, rows, cols);
    int blockSize = format.blockSize;
    for (int b = 0; b < batch; b++) {
      for (int row = 0; row < rows; row++) {
        double exact = 0.0;
        double l1 = 0.0;
        double quantBound = 0.0;
        for (int block = 0; block < cols / blockSize; block++) {
          double max = 0.0;
          for (int i = 0; i < blockSize; i++) {
            max = Math.max(max, Math.abs(queries[b * cols + block * blockSize + i]));
          }
          for (int i = 0; i < blockSize; i++) {
            int col = block * blockSize + i;
            double term = w[row * cols + col] * queries[b * cols + col];
            exact += term;
            l1 += Math.abs(term);
            quantBound += Math.abs(w[row * cols + col]) * max / 127 / 2;
          }
        }
        double bandTolerance = BAND_RELATIVE_TOLERANCE * l1 + ABSOLUTE_SLACK;
        double tolerance = band ? bandTolerance : quantBound * (1 + 1e-3) + bandTolerance;
        assertThat(Math.abs(out[b * rows + row] - exact))
            .as("%s batch=%d row=%d %s", format, b, row, band ? "band" : "integer")
            .isLessThanOrEqualTo(tolerance);
      }
    }
  }

  static String readAll(Process process) throws IOException {
    return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
  }
}
