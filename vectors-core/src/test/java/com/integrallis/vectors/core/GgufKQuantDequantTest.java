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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Bit-identity of the Vector API K-quant dequantisation arms ({@link GgufKQuantDequant}) against
 * the scalar reference loops in {@link GgufBandGemm}. The pre-registered rule
 * (vectors-bench/jmh-results/2026-09-17-kquant-dequant) requires {@code Float.floatToIntBits}
 * equality: every arm computes the same IEEE float32 expression per element ({@code scale * q -
 * min} for Q4_K, {@code scale * (q - 32)} for Q6_K), so there is no tolerance.
 */
class GgufKQuantDequantTest {

  static final int Q4_K_BYTES = VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES;
  static final int Q6_K_BYTES = VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES;

  /** float16 bit patterns that stress the scale arithmetic. */
  static final short[] EDGE_FLOAT16 = {
    0x0000, // +0
    (short) 0x8000, // -0
    0x0001, // smallest positive subnormal
    (short) 0x8001, // smallest negative subnormal
    0x03FF, // largest subnormal
    0x0400, // smallest normal
    0x3C00, // 1.0
    (short) 0xBC00, // -1.0
    0x7BFF, // 65504, max finite
    (short) 0xFBFF, // -65504
    0x7C00, // +inf
    (short) 0xFC00, // -inf
    0x7E00, // NaN
  };

  static Stream<Arguments> armsAndFormats() {
    Stream.Builder<Arguments> builder = Stream.builder();
    for (GgufKQuantDequant.Arm arm : GgufKQuantDequant.Arm.values()) {
      for (int format : new int[] {GgufBandGemm.Q4_K, GgufBandGemm.Q6_K}) {
        builder.add(Arguments.of(arm, format));
      }
    }
    return builder.build();
  }

  @ParameterizedTest(name = "{0} format={1}")
  @MethodSource("armsAndFormats")
  void everyArmIsBitIdenticalToScalarOnAdversarialAndRandomBlocks(
      GgufKQuantDequant.Arm arm, int format) {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "SIMD dequant requires the Vector API");
    byte[] blocks = testBlocks(format, new Random(1234L + format));
    String mismatch = firstMismatch(arm, format, blocks);
    assertThat(mismatch)
        .as(
            "%s (effective %s, lanes %d) vs scalar",
            arm, GgufKQuantDequant.effective(arm), GgufKQuantDequant.LANES)
        .isNull();
  }

  @ParameterizedTest(name = "{0} format={1}")
  @MethodSource("armsAndFormats")
  void multiBlockRunsHonourSourceOffsetAndDestinationOffset(GgufKQuantDequant.Arm arm, int format) {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "SIMD dequant requires the Vector API");
    int blockBytes = GgufBandGemm.blockBytes(format);
    byte[] blocks = testBlocks(format, new Random(99L + format));
    int total = blocks.length / blockBytes;
    MemorySegment weights = MemorySegment.ofArray(blocks);
    // Uneven windows: a leading offset, runs of 1, 2, 3 and 5 blocks, and a non-zero dst0.
    int[] runs = {1, 2, 3, 5};
    int block = 3;
    int run = 0;
    while (block < total) {
      int count = Math.min(runs[run++ % runs.length], total - block);
      int dst0 = 17 + run;
      float[] expected = new float[dst0 + count * 256 + 9];
      float[] actual = expected.clone();
      Arrays.fill(expected, -7.0f);
      Arrays.fill(actual, -7.0f);
      GgufBandGemm.dequantize(
          GgufKQuantDequant.Arm.SCALAR,
          format,
          weights,
          (long) block * blockBytes,
          count,
          new byte[count * blockBytes],
          expected,
          dst0);
      GgufBandGemm.dequantize(
          arm,
          format,
          weights,
          (long) block * blockBytes,
          count,
          new byte[count * blockBytes + 64],
          actual,
          dst0);
      assertThat(bits(actual)).as("block %d count %d", block, count).isEqualTo(bits(expected));
      block += count;
    }
  }

  @Test
  void scalarArmIsTheExistingReferenceLoops() {
    byte[] blocks = testBlocks(GgufBandGemm.Q4_K, new Random(3L));
    float[] viaArm = new float[256];
    float[] reference = new float[256];
    GgufKQuantDequant.q4_K(GgufKQuantDequant.Arm.SCALAR, blocks, 0, viaArm, 0);
    GgufBandGemm.dequantizeQ4_KBlock(blocks, 0, reference, 0);
    assertThat(bits(viaArm)).isEqualTo(bits(reference));
  }

  @Test
  void fusedArmsFallBackToSplitOnlyWithoutALaneMatchedByteSpecies() {
    assertThat(GgufKQuantDequant.effective(GgufKQuantDequant.Arm.SIMD_BYTE, 4))
        .isEqualTo(GgufKQuantDequant.Arm.SIMD_SPLIT);
    assertThat(GgufKQuantDequant.effective(GgufKQuantDequant.Arm.SIMD_INT, 4))
        .isEqualTo(GgufKQuantDequant.Arm.SIMD_SPLIT);
    for (int lanes : new int[] {8, 16}) {
      for (GgufKQuantDequant.Arm arm : GgufKQuantDequant.Arm.values()) {
        assertThat(GgufKQuantDequant.effective(arm, lanes)).isEqualTo(arm);
      }
    }
    assertThat(GgufKQuantDequant.effective(GgufKQuantDequant.Arm.SCALAR, 4))
        .isEqualTo(GgufKQuantDequant.Arm.SCALAR);
    assertThat(GgufKQuantDequant.effective(GgufKQuantDequant.Arm.SIMD_SPLIT, 4))
        .isEqualTo(GgufKQuantDequant.Arm.SIMD_SPLIT);
  }

  @Test
  void propertyParsingAndDefault() {
    assertThat(System.getProperty(GgufKQuantDequant.PROPERTY)).isNull();
    assertThat(GgufKQuantDequant.REQUESTED).isEqualTo(GgufKQuantDequant.Arm.SCALAR);
    assertThat(GgufKQuantDequant.parse(null)).isEqualTo(GgufKQuantDequant.Arm.SCALAR);
    assertThat(GgufKQuantDequant.parse(" SIMD-int ")).isEqualTo(GgufKQuantDequant.Arm.SIMD_INT);
    assertThat(GgufKQuantDequant.parse("simd-byte")).isEqualTo(GgufKQuantDequant.Arm.SIMD_BYTE);
    assertThat(GgufKQuantDequant.parse("simd-split")).isEqualTo(GgufKQuantDequant.Arm.SIMD_SPLIT);
    assertThatIllegalArgumentException().isThrownBy(() -> GgufKQuantDequant.parse("simd"));
    assertThat(GgufBatchedMatmulKernel.bandDequantConfiguration())
        .isEqualTo(
            "kquant-dequant(requested=scalar,effective=scalar,lanes="
                + GgufKQuantDequant.LANES
                + ")");
  }

  static Stream<Arguments> freshJvmCases() {
    List<Arguments> cases = new ArrayList<>();
    List<Integer> widths = new ArrayList<>(List.of(128, 256));
    if (PanamaConstants.PREFERRED_BITS >= 512) {
      widths.add(512);
    }
    for (int maxBits : widths) {
      for (String arm : new String[] {"scalar", "simd-byte", "simd-int", "simd-split"}) {
        cases.add(Arguments.of(maxBits, arm));
      }
    }
    return cases.stream();
  }

  @ParameterizedTest(name = "maxBits={0} dequant={1}")
  @MethodSource("freshJvmCases")
  void propertySelectsArmAndItIsBitIdenticalInAFreshJvm(int maxBits, String arm) throws Exception {
    assumeTrue(VectorizationProvider.isPanamaEnabled(), "SIMD dequant requires the Vector API");
    assumeTrue(maxBits <= PanamaConstants.PREFERRED_BITS, "host lacks this width");
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Dvectors.maxBits=" + maxBits,
                "-D" + GgufKQuantDequant.PROPERTY + "=" + arm,
                "-D" + GgufKQuantDequant.COUNT_PROPERTY + "=true",
                "-cp",
                System.getProperty("java.class.path"),
                GgufKQuantDequantProbe.class.getName())
            .redirectErrorStream(true)
            .start();
    boolean completed = process.waitFor(180, TimeUnit.SECONDS);
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    assertThat(completed).as("probe timed out; output:%n%s", output).isTrue();
    assertThat(process.exitValue()).as("probe output:%n%s", output).isZero();
    int lanes = maxBits / Float.SIZE;
    String effective = lanes == 4 && arm.matches("simd-(byte|int)") ? "simd-split" : arm;
    assertThat(output)
        .contains(
            "bandDequant=kquant-dequant(requested="
                + arm
                + ",effective="
                + effective
                + ",lanes="
                + lanes
                + ")")
        .contains("probe ok");
  }

  // ---- shared with the probe ----

  /** Returns a description of the first element that differs from scalar, or null. */
  static String firstMismatch(GgufKQuantDequant.Arm arm, int format, byte[] blocks) {
    int blockBytes = GgufBandGemm.blockBytes(format);
    float[] expected = new float[256];
    float[] actual = new float[256];
    for (int block = 0; block * blockBytes < blocks.length; block++) {
      int bo = block * blockBytes;
      Arrays.fill(actual, Float.MIN_VALUE);
      if (format == GgufBandGemm.Q4_K) {
        GgufBandGemm.dequantizeQ4_KBlock(blocks, bo, expected, 0);
        GgufKQuantDequant.q4_K(arm, blocks, bo, actual, 0);
      } else {
        GgufBandGemm.dequantizeQ6_KBlock(blocks, bo, expected, 0);
        GgufKQuantDequant.q6_K(arm, blocks, bo, actual, 0);
      }
      for (int i = 0; i < 256; i++) {
        if (Float.floatToIntBits(expected[i]) != Float.floatToIntBits(actual[i])) {
          return "block "
              + block
              + " element "
              + i
              + ": scalar "
              + expected[i]
              + " "
              + arm
              + " "
              + actual[i];
        }
      }
    }
    return null;
  }

  /** Adversarial blocks first, then fully random blocks, then realistic random blocks. */
  static byte[] testBlocks(int format, Random random) {
    List<byte[]> blocks =
        format == GgufBandGemm.Q4_K ? adversarialQ4_K(random) : adversarialQ6_K(random);
    int blockBytes = GgufBandGemm.blockBytes(format);
    for (int i = 0; i < 400; i++) {
      byte[] block = new byte[blockBytes];
      random.nextBytes(block);
      if (i % 2 == 1) {
        short d = Float.floatToFloat16(0.0005f + random.nextFloat() * 0.02f);
        if (format == GgufBandGemm.Q4_K) {
          putFloat16(block, 0, d);
          putFloat16(block, 2, Float.floatToFloat16((random.nextFloat() - 0.25f) * 0.01f));
        } else {
          putFloat16(block, 208, d);
        }
      }
      blocks.add(block);
    }
    byte[] all = new byte[blocks.size() * blockBytes];
    for (int i = 0; i < blocks.size(); i++) {
      System.arraycopy(blocks.get(i), 0, all, i * blockBytes, blockBytes);
    }
    return all;
  }

  private static List<byte[]> adversarialQ4_K(Random random) {
    List<byte[]> out = new ArrayList<>();
    out.add(new byte[Q4_K_BYTES]);
    out.add(filled(Q4_K_BYTES, (byte) 0xFF));
    for (short d : EDGE_FLOAT16) {
      for (short dMin : EDGE_FLOAT16) {
        byte[] block = new byte[Q4_K_BYTES];
        random.nextBytes(block);
        putFloat16(block, 0, d);
        putFloat16(block, 2, dMin);
        out.add(block);
      }
    }
    // Maximum 6-bit scales and mins in both packings, with extreme and ordinary d/dmin.
    for (short[] dd :
        new short[][] {{0x3C00, 0x3C00}, {0x7BFF, (short) 0xFBFF}, {(short) 0xFBFF, 0x7BFF}}) {
      for (int quantPattern : new int[] {0x00, 0xFF, 0x0F, 0xF0, 0x5A, -1}) {
        byte[] block = new byte[Q4_K_BYTES];
        putFloat16(block, 0, dd[0]);
        putFloat16(block, 2, dd[1]);
        Arrays.fill(block, 4, 16, (byte) 0xFF);
        fillQuants(block, 16, 144, quantPattern, random);
        out.add(block);
      }
    }
    // Scales at 63 with mins at 0, and mins at 63 with scales at 0 (both packings).
    byte[] scalesOnly = new byte[Q4_K_BYTES];
    putFloat16(scalesOnly, 0, (short) 0x3C00);
    putFloat16(scalesOnly, 2, (short) 0xBC00);
    for (int j = 0; j < 4; j++) {
      scalesOnly[4 + j] = 0x3F; // low 6 bits scale 63, top bits feed high groups' scales
      scalesOnly[4 + j + 4] = (byte) 0xC0; // min 0 low, top bits feed high groups' mins
      scalesOnly[4 + j + 8] = 0x0F; // high groups: scale low nibble 15, min high nibble 0
    }
    fillQuants(scalesOnly, 16, 144, -1, random);
    out.add(scalesOnly);
    byte[] minsOnly = new byte[Q4_K_BYTES];
    putFloat16(minsOnly, 0, (short) 0xBC00);
    putFloat16(minsOnly, 2, (short) 0xBC00); // negative dmin: subtracting a negative min
    for (int j = 0; j < 4; j++) {
      minsOnly[4 + j] = (byte) 0xC0;
      minsOnly[4 + j + 4] = 0x3F;
      minsOnly[4 + j + 8] = (byte) 0xF0;
    }
    fillQuants(minsOnly, 16, 144, -1, random);
    out.add(minsOnly);
    return out;
  }

  private static List<byte[]> adversarialQ6_K(Random random) {
    List<byte[]> out = new ArrayList<>();
    out.add(new byte[Q6_K_BYTES]);
    out.add(filled(Q6_K_BYTES, (byte) 0xFF));
    for (short d : EDGE_FLOAT16) {
      for (int scalePattern : new int[] {127, -128, 0, 1, -1, -2}) {
        for (int quantPattern : new int[] {0x00, 0xFF, 0xAA, 0x55, -1}) {
          byte[] block = new byte[Q6_K_BYTES];
          fillQuants(block, 0, 192, quantPattern, random);
          if (scalePattern == -2) {
            random.nextBytes(block);
          } else {
            Arrays.fill(block, 192, 208, (byte) scalePattern);
          }
          putFloat16(block, 208, d);
          out.add(block);
        }
      }
    }
    // Alternating extreme sub-block scales so every sub-block boundary is exercised.
    byte[] alternating = new byte[Q6_K_BYTES];
    fillQuants(alternating, 0, 192, -1, random);
    for (int s = 0; s < 16; s++) {
      alternating[192 + s] = (byte) (s % 2 == 0 ? 127 : -128);
    }
    putFloat16(alternating, 208, (short) 0x7BFF);
    out.add(alternating);
    return out;
  }

  private static void fillQuants(byte[] block, int from, int to, int pattern, Random random) {
    if (pattern < 0) {
      byte[] tmp = new byte[to - from];
      random.nextBytes(tmp);
      System.arraycopy(tmp, 0, block, from, tmp.length);
    } else {
      Arrays.fill(block, from, to, (byte) pattern);
    }
  }

  private static byte[] filled(int size, byte value) {
    byte[] block = new byte[size];
    Arrays.fill(block, value);
    return block;
  }

  private static void putFloat16(byte[] block, int offset, short value) {
    ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).putShort(offset, value);
  }

  static int[] bits(float[] values) {
    int[] out = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = Float.floatToIntBits(values[i]);
    }
    return out;
  }
}
