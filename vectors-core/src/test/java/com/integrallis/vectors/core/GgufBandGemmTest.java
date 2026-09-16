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

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Correctness of the experimental band F32 GEMM against an exact double-precision reference.
 *
 * <p><b>Tolerances.</b> The band arm computes {@code sum_k w[k] * x[k]} in float32, so it is
 * compared with the same sum evaluated in double over independently dequantized weights. Accepted
 * error: {@code |band - exact| <= 1e-4 * sum_k |w[k] * x[k]| + 1e-6}. Relative to the magnitude of
 * the terms (not the result, which can cancel towards zero), float32 rounding in recursive lane
 * accumulation is bounded to first order by about {@code (k / lanes) * eps} (2.4e-4 for k = 8192 on
 * 4 lanes, eps = 1.19e-7) and in practice sits near {@code sqrt(k) * eps}; 1e-4 is tight enough to
 * catch any indexing or bit-extraction error, which moves a result by whole terms.
 *
 * <p>The existing integer kernels cannot be compared bit-for-bit: they quantize each activation
 * block to int8 first. Their deviation from the exact result is bounded analytically by the
 * activation rounding error, {@code sum_k |w[k]| * scale(block(k)) / 2}, where {@code scale =
 * max|x| / 127} per Q8_K (256) or Q8_0 (32) block. Checking both the Panama and the scalar
 * reference provider against that bound validates the test's dequantization against the shipped
 * kernels' reading of the bit layout, and bounds {@code |band - integer|} by the sum of both
 * tolerances.
 */
class GgufBandGemmTest {

  private static final double BAND_RELATIVE_TOLERANCE = 1e-4;
  private static final double ABSOLUTE_SLACK = 1e-6;

  private static double maxBandRelativeError;
  private static double maxIntegerRelativeError;
  private static double maxIntegerBoundUse;

  enum Format {
    Q4_K(256, 144),
    Q6_K(256, 210),
    Q8_0(32, 34);

    final int blockSize;
    final int blockBytes;

    Format(int blockSize, int blockBytes) {
      this.blockSize = blockSize;
      this.blockBytes = blockBytes;
    }
  }

  static Stream<Arguments> shapes() {
    // {batch, rows, cols}: decode (batch 1), prefill, batch not a multiple of 3 or 4, rows not a
    // multiple of the band, cols below / equal to / not a multiple of the 512 column block,
    // multiple panels, and one parallel-threshold shape per format.
    int[][] kQuant = {
      {1, 1, 256},
      {1, 7, 256},
      {2, 5, 512},
      {3, 3, 768},
      {4, 97, 768},
      {5, 13, 2560},
      {7, 200, 1280},
      {11, 64, 256},
      {1, 130, 2560},
      {5, 512, 2560}
    };
    int[][] q8 = {
      {1, 1, 32},
      {2, 3, 96},
      {5, 7, 544},
      {4, 97, 1024},
      {7, 33, 2048},
      {1, 200, 2560},
      {3, 1600, 2560}
    };
    Stream.Builder<Arguments> builder = Stream.builder();
    for (Format format : new Format[] {Format.Q4_K, Format.Q6_K}) {
      for (int[] shape : kQuant) {
        builder.add(Arguments.of(format, shape[0], shape[1], shape[2]));
      }
    }
    for (int[] shape : q8) {
      builder.add(Arguments.of(Format.Q8_0, shape[0], shape[1], shape[2]));
    }
    return builder.build();
  }

  @ParameterizedTest(name = "{0} batch={1} rows={2} cols={3}")
  @MethodSource("shapes")
  void bandMatchesExactReferenceAndIntegerKernelsWithinBounds(
      Format format, int batch, int rows, int cols) {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    Random random = new Random(31L * format.ordinal() + 7L * batch + rows * 131L + cols);
    MemorySegment weights = MemorySegment.ofArray(randomBlocks(format, random, rows, cols));
    float[] queries = new float[batch * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = (float) random.nextGaussian();
    }

    double[] dequantized = dequantizeReference(format, weights, rows, cols);
    float[] band = run(format, VectorUtilSupportKind.BAND, weights, queries, batch, rows, cols);
    float[] integer =
        run(format, VectorUtilSupportKind.PANAMA_INTEGER, weights, queries, batch, rows, cols);
    float[] scalar =
        run(format, VectorUtilSupportKind.SCALAR_INTEGER, weights, queries, batch, rows, cols);

    int blockSize = format == Format.Q8_0 ? 32 : 256;
    boolean anyDifference = false;
    for (int b = 0; b < batch; b++) {
      double[] activationHalfStep = new double[cols];
      for (int block = 0; block < cols / blockSize; block++) {
        double max = 0.0;
        for (int i = 0; i < blockSize; i++) {
          max = Math.max(max, Math.abs(queries[b * cols + block * blockSize + i]));
        }
        Arrays.fill(activationHalfStep, block * blockSize, (block + 1) * blockSize, max / 127 / 2);
      }
      for (int row = 0; row < rows; row++) {
        double exact = 0.0;
        double l1 = 0.0;
        double quantBound = 0.0;
        for (int col = 0; col < cols; col++) {
          double w = dequantized[row * cols + col];
          double x = queries[b * cols + col];
          exact += w * x;
          l1 += Math.abs(w * x);
          quantBound += Math.abs(w) * activationHalfStep[col];
        }
        int index = b * rows + row;
        double bandError = Math.abs(band[index] - exact);
        double bandTolerance = BAND_RELATIVE_TOLERANCE * l1 + ABSOLUTE_SLACK;
        assertThat(bandError)
            .as("band %s out[%d] band=%s exact=%s l1=%s", format, index, band[index], exact, l1)
            .isLessThanOrEqualTo(bandTolerance);

        double integerTolerance = quantBound * (1 + 1e-3) + bandTolerance;
        for (float[] candidate : new float[][] {integer, scalar}) {
          double integerError = Math.abs(candidate[index] - exact);
          assertThat(integerError)
              .as("integer %s out[%d] value=%s exact=%s", format, index, candidate[index], exact)
              .isLessThanOrEqualTo(integerTolerance);
          synchronized (GgufBandGemmTest.class) {
            maxIntegerRelativeError = Math.max(maxIntegerRelativeError, integerError / l1);
            maxIntegerBoundUse = Math.max(maxIntegerBoundUse, integerError / integerTolerance);
          }
        }
        assertThat((double) Math.abs(band[index] - integer[index]))
            .isLessThanOrEqualTo(integerTolerance + bandTolerance);
        anyDifference |= band[index] != integer[index];
        synchronized (GgufBandGemmTest.class) {
          maxBandRelativeError = Math.max(maxBandRelativeError, bandError / l1);
        }
      }
    }
    // Observability: the band arm must not silently be the integer kernel.
    assertThat(anyDifference).as("band output identical to integer output").isTrue();
  }

  @Test
  void defaultEntryPointRunsTheReportedKernel() {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    assertThat(System.getProperty(GgufBatchedMatmulKernel.PROPERTY)).isNull();
    assertThat(VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel()).isEqualTo("integer");
    Random random = new Random(5L);
    int batch = 3;
    int rows = 9;
    int cols = 512;
    MemorySegment weights = MemorySegment.ofArray(randomBlocks(Format.Q4_K, random, rows, cols));
    float[] queries = new float[batch * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = (float) random.nextGaussian();
    }
    float[] defaultOut = new float[batch * rows];
    VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
        queries,
        weights,
        batch,
        rows,
        cols,
        defaultOut,
        new byte[batch * cols],
        new float[batch * 2],
        new short[batch * 32]);
    assertThat(defaultOut)
        .containsExactly(
            run(
                Format.Q4_K,
                VectorUtilSupportKind.PANAMA_INTEGER,
                weights,
                queries,
                batch,
                rows,
                cols));
  }

  @ParameterizedTest(name = "maxBits={0} tile={1}")
  @org.junit.jupiter.params.provider.CsvSource({"256, 4x4", "128, 3x3", "128, 4x4"})
  void propertySelectsBandAndCapabilitiesReportItInAFreshJvm(int maxBits, String tile)
      throws Exception {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "band arm requires the Vector API");
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Dvectors.maxBits=" + maxBits,
                "-D" + GgufBatchedMatmulKernel.PROPERTY + "=band",
                "-D" + GgufBandGemm.TILE_PROPERTY + "=" + tile,
                "-cp",
                System.getProperty("java.class.path"),
                GgufBandGemmSwitchProbe.class.getName())
            .redirectErrorStream(true)
            .start();
    boolean completed = process.waitFor(120, TimeUnit.SECONDS);
    String output = readOutput(process);
    assertThat(completed).as("probe timed out; output:%n%s", output).isTrue();
    assertThat(process.exitValue()).as("probe output:%n%s", output).isZero();
    int lanes = Math.min(maxBits, PanamaConstants.PREFERRED_BITS) / Float.SIZE;
    assertThat(output)
        .contains(
            "ggufBatchedMatmulKernel=band-f32(tile="
                + tile
                + ",kc=512,panelKb=256,lanes="
                + lanes
                + ")");
  }

  /**
   * Largest {@code |band - exact| / tolerance} over every output of one random problem (a value
   * above 1 is a failure). Used by the fresh-JVM probe to cover other tiles and lane counts.
   */
  static double bandToleranceUse(Format format, int batch, int rows, int cols) {
    Random random = new Random(97L * format.ordinal() + batch + rows * 7L + cols);
    MemorySegment weights = MemorySegment.ofArray(randomBlocks(format, random, rows, cols));
    float[] queries = new float[batch * cols];
    for (int i = 0; i < queries.length; i++) {
      queries[i] = (float) random.nextGaussian();
    }
    double[] dequantized = dequantizeReference(format, weights, rows, cols);
    float[] band = run(format, VectorUtilSupportKind.BAND, weights, queries, batch, rows, cols);
    double worst = 0.0;
    for (int b = 0; b < batch; b++) {
      for (int row = 0; row < rows; row++) {
        double exact = 0.0;
        double l1 = 0.0;
        for (int col = 0; col < cols; col++) {
          double term = dequantized[row * cols + col] * queries[b * cols + col];
          exact += term;
          l1 += Math.abs(term);
        }
        double tolerance = BAND_RELATIVE_TOLERANCE * l1 + ABSOLUTE_SLACK;
        worst = Math.max(worst, Math.abs(band[b * rows + row] - exact) / tolerance);
      }
    }
    return worst;
  }

  @Test
  void rejectsUnknownKernelNames() {
    assertThat(GgufBatchedMatmulKernel.parse(null)).isEqualTo(GgufBatchedMatmulKernel.INTEGER);
    assertThat(GgufBatchedMatmulKernel.parse(" Band ")).isEqualTo(GgufBatchedMatmulKernel.BAND_F32);
    org.assertj.core.api.Assertions.assertThatIllegalArgumentException()
        .isThrownBy(() -> GgufBatchedMatmulKernel.parse("f16"));
    org.assertj.core.api.Assertions.assertThatIllegalArgumentException()
        .isThrownBy(() -> GgufBandGemm.resolveKc("300"));
  }

  @AfterAll
  static void report() {
    // Recorded in the experiment README; not an assertion.
    System.out.printf(
        "band-gemm tolerance report: max |band-exact|/L1 = %.3e (limit %.1e); "
            + "max |integer-exact|/L1 = %.3e; max integer error / analytic bound = %.3f%n",
        maxBandRelativeError, BAND_RELATIVE_TOLERANCE, maxIntegerRelativeError, maxIntegerBoundUse);
  }

  enum VectorUtilSupportKind {
    BAND,
    PANAMA_INTEGER,
    SCALAR_INTEGER
  }

  static float[] run(
      Format format,
      VectorUtilSupportKind kind,
      MemorySegment weights,
      float[] queries,
      int batch,
      int rows,
      int cols) {
    float[] out = new float[batch * rows];
    Arrays.fill(out, Float.NaN);
    byte[] quants = new byte[batch * cols];
    float[] scales = new float[batch * (cols / format.blockSize)];
    short[] sums = new short[batch * (cols / 16)];
    GgufBatchedMatmulKernel kernel =
        kind == VectorUtilSupportKind.BAND
            ? GgufBatchedMatmulKernel.BAND_F32
            : GgufBatchedMatmulKernel.INTEGER;
    if (kind == VectorUtilSupportKind.SCALAR_INTEGER) {
      VectorUtilSupport scalar = VectorizationProvider.newScalarProvider();
      switch (format) {
        case Q4_K ->
            scalar.ggufQ4_KQ8_KBatchedMatmul(
                queries, weights, batch, rows, cols, out, quants, scales, sums);
        case Q6_K ->
            scalar.ggufQ6_KQ8_KBatchedMatmul(
                queries, weights, batch, rows, cols, out, quants, scales);
        case Q8_0 ->
            scalar.ggufQ8_0Q8_0BatchedMatmul(
                queries, weights, batch, rows, cols, out, quants, scales);
      }
      return out;
    }
    switch (format) {
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
              queries, weights, batch, rows, cols, out, quants, scales, sums, kernel);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              queries, weights, batch, rows, cols, out, quants, scales, kernel);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              queries, weights, batch, rows, cols, out, quants, scales, kernel);
    }
    return out;
  }

  static byte[] randomBlocks(Format format, Random random, int rows, int cols) {
    int blocks = rows * (cols / format.blockSize);
    byte[] bytes = new byte[blocks * format.blockBytes];
    random.nextBytes(bytes);
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < blocks; block++) {
      int offset = block * format.blockBytes;
      short d = Float.floatToFloat16(0.0005f + random.nextFloat() * 0.02f);
      switch (format) {
        case Q4_K -> {
          buffer.putShort(offset, d);
          buffer.putShort(offset + 2, Float.floatToFloat16(random.nextFloat() * 0.01f));
        }
        case Q6_K -> buffer.putShort(offset + 208, d);
        case Q8_0 -> buffer.putShort(offset, d);
      }
    }
    return bytes;
  }

  /** Element-wise dequantization written independently of both kernels. */
  static double[] dequantizeReference(Format format, MemorySegment weights, int rows, int cols) {
    double[] out = new double[rows * cols];
    int blocksPerRow = cols / format.blockSize;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        long b = ((long) row * blocksPerRow + col / format.blockSize) * format.blockBytes;
        int e = col % format.blockSize;
        out[row * cols + col] =
            switch (format) {
              case Q4_K -> {
                int group = e / 32;
                int packed =
                    weights.get(ValueLayout.JAVA_BYTE, b + 16 + (group >> 1) * 32L + e % 32) & 0xFF;
                int nibble = (packed >> ((group & 1) * 4)) & 0x0F;
                double d = float16(weights, b);
                double dMin = float16(weights, b + 2);
                yield d * GgufQuantizationSupport.qKScale(weights, b + 4, group) * nibble
                    - dMin * GgufQuantizationSupport.qKMin(weights, b + 4, group);
              }
              case Q6_K -> {
                int half = e / 128;
                int quarter = (e % 128) / 32;
                int index = (e % 128) % 32;
                int ql =
                    weights.get(ValueLayout.JAVA_BYTE, b + half * 64L + (quarter & 1) * 32L + index)
                        & 0xFF;
                int qh = weights.get(ValueLayout.JAVA_BYTE, b + 128 + half * 32L + index) & 0xFF;
                int quant =
                    ((ql >> ((quarter >> 1) * 4)) & 0x0F) | (((qh >> (2 * quarter)) & 3) << 4);
                int scale =
                    weights.get(
                        ValueLayout.JAVA_BYTE, b + 192 + half * 8L + 2L * quarter + index / 16);
                yield float16(weights, b + 208) * scale * (quant - 32);
              }
              case Q8_0 -> float16(weights, b) * weights.get(ValueLayout.JAVA_BYTE, b + 2 + e);
            };
      }
    }
    return out;
  }

  private static double float16(MemorySegment segment, long offset) {
    return Float.float16ToFloat(
        segment.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), offset));
  }

  private static String readOutput(Process process) throws IOException {
    return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
  }
}
