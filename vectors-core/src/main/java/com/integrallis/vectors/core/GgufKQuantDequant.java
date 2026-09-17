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

import java.util.Locale;
import java.util.concurrent.atomic.LongAdder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Experimental Vector API dequantisation of GGUF Q4_K and Q6_K super-blocks (256 elements) for the
 * band GEMM arm. Every arm evaluates exactly the scalar reference's float32 expression per element
 * ({@code scale * q - min} for Q4_K, {@code scale * (q - 32)} for Q6_K, with the same scalar {@code
 * scale}/{@code min}), so the outputs are bit-identical; only the extraction of {@code q} and the
 * lane width differ.
 *
 * <ul>
 *   <li>{@code scalar}: {@link GgufBandGemm#dequantizeQ4_KBlock} / {@link
 *       GgufBandGemm#dequantizeQ6_KBlock}, the reference path.
 *   <li>{@code simd-byte}: fused; load {@code L} packed bytes ({@code L} = float lanes), extract in
 *       byte lanes, {@code B2F}, scale in float lanes.
 *   <li>{@code simd-int}: fused; load {@code L} bytes, {@code B2I}, extract in int lanes, {@code
 *       I2F}, scale in float lanes.
 *   <li>{@code simd-split}: unpack a group of 32 (Q4_K) or a quarter of 32 (Q6_K) into signed quant
 *       bytes with wide byte lanes, then {@code B2F} and scale in {@code L} float lanes, or in a
 *       scalar loop when no {@code L}-lane byte species exists.
 * </ul>
 *
 * <p>The fused arms need a byte species with exactly {@code L} lanes: 8 (64-bit) or 16 (128-bit).
 * With 128-bit float species ({@code L} = 4) they resolve to {@code simd-split}; {@link
 * #describe()} reports requested and effective arm. Select with {@code
 * -Dvectors.gguf.band.dequant=scalar|simd-byte|simd-int|simd-split}.
 */
final class GgufKQuantDequant {

  static final String PROPERTY = "vectors.gguf.band.dequant";

  /** Test-only observability: count dequantize calls per arm. Folds away when false. */
  static final String COUNT_PROPERTY = "vectors.gguf.band.dequant.count";

  enum Arm {
    SCALAR("scalar"),
    SIMD_BYTE("simd-byte"),
    SIMD_INT("simd-int"),
    SIMD_SPLIT("simd-split");

    final String label;

    Arm(String label) {
      this.label = label;
    }
  }

  static final VectorSpecies<Float> FLOAT_SPECIES = PanamaVectorUtilSupport.FLOAT_SPECIES;
  static final int LANES = FLOAT_SPECIES.length();

  /**
   * Byte and int species with exactly {@link #LANES} lanes (meaningful only when fused-capable).
   */
  static final VectorSpecies<Byte> LANE_BYTE_SPECIES =
      LANES >= 16 ? ByteVector.SPECIES_128 : ByteVector.SPECIES_64;

  static final VectorSpecies<Integer> LANE_INT_SPECIES =
      LANES >= 16 ? IntVector.SPECIES_512 : IntVector.SPECIES_256;

  /** Wide byte species for the split arm's unpack stage: 32 lanes where 256-bit is available. */
  static final VectorSpecies<Byte> WIDE_BYTE_SPECIES =
      FLOAT_SPECIES.vectorBitSize() >= 256 ? ByteVector.SPECIES_256 : ByteVector.SPECIES_128;

  static final boolean LANE_MATCHED = hasLaneMatchedByteSpecies(LANES);

  static final Arm REQUESTED = parse(System.getProperty(PROPERTY));
  static final Arm ACTIVE = effective(REQUESTED);

  static final boolean COUNTING = Boolean.getBoolean(COUNT_PROPERTY);
  private static final LongAdder[] CALLS = {
    new LongAdder(), new LongAdder(), new LongAdder(), new LongAdder()
  };

  private static final ThreadLocal<byte[]> UNPACKED = ThreadLocal.withInitial(() -> new byte[256]);

  private GgufKQuantDequant() {}

  static String describe() {
    return "kquant-dequant(requested="
        + REQUESTED.label
        + ",effective="
        + ACTIVE.label
        + ",lanes="
        + LANES
        + ")";
  }

  static Arm parse(String configured) {
    if (configured == null || configured.isBlank()) {
      return Arm.SCALAR;
    }
    String value = configured.trim().toLowerCase(Locale.ROOT);
    for (Arm arm : Arm.values()) {
      if (arm.label.equals(value)) {
        return arm;
      }
    }
    throw new IllegalArgumentException(
        "-D" + PROPERTY + " must be scalar, simd-byte, simd-int or simd-split; got: " + configured);
  }

  static Arm effective(Arm requested) {
    return effective(requested, LANES);
  }

  static Arm effective(Arm requested, int lanes) {
    boolean fused = requested == Arm.SIMD_BYTE || requested == Arm.SIMD_INT;
    return fused && !hasLaneMatchedByteSpecies(lanes) ? Arm.SIMD_SPLIT : requested;
  }

  static boolean hasLaneMatchedByteSpecies(int lanes) {
    return lanes == 8 || lanes == 16;
  }

  static void count(Arm arm) {
    CALLS[arm.ordinal()].increment();
  }

  static long calls(Arm arm) {
    return CALLS[arm.ordinal()].sum();
  }

  // ---- block runs: the arm switch sits outside the per-block loop ----

  static void q4_KBlocks(Arm arm, byte[] raw, int blocks, float[] dst, int dst0) {
    int blockBytes = VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES;
    switch (effective(arm)) {
      case SCALAR -> {
        for (int block = 0; block < blocks; block++) {
          GgufBandGemm.dequantizeQ4_KBlock(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_BYTE -> {
        for (int block = 0; block < blocks; block++) {
          q4_KByte(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_INT -> {
        for (int block = 0; block < blocks; block++) {
          q4_KInt(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_SPLIT -> {
        byte[] unpacked = UNPACKED.get();
        for (int block = 0; block < blocks; block++) {
          q4_KSplit(raw, block * blockBytes, dst, dst0 + block * 256, unpacked);
        }
      }
    }
  }

  static void q6_KBlocks(Arm arm, byte[] raw, int blocks, float[] dst, int dst0) {
    int blockBytes = VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES;
    switch (effective(arm)) {
      case SCALAR -> {
        for (int block = 0; block < blocks; block++) {
          GgufBandGemm.dequantizeQ6_KBlock(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_BYTE -> {
        for (int block = 0; block < blocks; block++) {
          q6_KByte(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_INT -> {
        for (int block = 0; block < blocks; block++) {
          q6_KInt(raw, block * blockBytes, dst, dst0 + block * 256);
        }
      }
      case SIMD_SPLIT -> {
        byte[] unpacked = UNPACKED.get();
        for (int block = 0; block < blocks; block++) {
          q6_KSplit(raw, block * blockBytes, dst, dst0 + block * 256, unpacked);
        }
      }
    }
  }

  /** One Q4_K block through {@code arm} (tests). */
  static void q4_K(Arm arm, byte[] raw, int bo, float[] dst, int o) {
    byte[] block =
        java.util.Arrays.copyOfRange(raw, bo, bo + VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    q4_KBlocks(arm, block, 1, dst, o);
  }

  /** One Q6_K block through {@code arm} (tests). */
  static void q6_K(Arm arm, byte[] raw, int bo, float[] dst, int o) {
    byte[] block =
        java.util.Arrays.copyOfRange(raw, bo, bo + VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    q6_KBlocks(arm, block, 1, dst, o);
  }

  // ---- Q4_K: d, dmin (float16), 12 bytes of 6-bit scales/mins, 128 bytes of nibbles ----

  static void q4_KByte(byte[] raw, int bo, float[] dst, int o) {
    float d = float16(raw, bo);
    float dMin = float16(raw, bo + 2);
    int scales = bo + VectorUtilSupport.GGUF_Q4_K_SCALES_OFFSET;
    int quants = bo + VectorUtilSupport.GGUF_Q4_K_QUANTS_OFFSET;
    int bound = LANE_BYTE_SPECIES.loopBound(32);
    for (int pair = 0; pair < 4; pair++) {
      int q = quants + pair * 32;
      int lowGroup = 2 * pair;
      int highGroup = lowGroup + 1;
      float lowScale = d * qKScale(raw, scales, lowGroup);
      float lowMin = dMin * qKMin(raw, scales, lowGroup);
      float highScale = d * qKScale(raw, scales, highGroup);
      float highMin = dMin * qKMin(raw, scales, highGroup);
      int lowOut = o + lowGroup * 32;
      int highOut = o + highGroup * 32;
      int i = 0;
      for (; i < bound; i += LANES) {
        ByteVector packed = ByteVector.fromArray(LANE_BYTE_SPECIES, raw, q + i);
        ((FloatVector) packed.and((byte) 0x0F).convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0))
            .mul(lowScale)
            .sub(lowMin)
            .intoArray(dst, lowOut + i);
        ((FloatVector)
                packed
                    .lanewise(VectorOperators.LSHR, 4)
                    .convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0))
            .mul(highScale)
            .sub(highMin)
            .intoArray(dst, highOut + i);
      }
      for (; i < 32; i++) {
        dst[lowOut + i] = lowScale * (raw[q + i] & 0x0F) - lowMin;
        dst[highOut + i] = highScale * ((raw[q + i] >> 4) & 0x0F) - highMin;
      }
    }
  }

  static void q4_KInt(byte[] raw, int bo, float[] dst, int o) {
    float d = float16(raw, bo);
    float dMin = float16(raw, bo + 2);
    int scales = bo + VectorUtilSupport.GGUF_Q4_K_SCALES_OFFSET;
    int quants = bo + VectorUtilSupport.GGUF_Q4_K_QUANTS_OFFSET;
    int bound = LANE_BYTE_SPECIES.loopBound(32);
    for (int pair = 0; pair < 4; pair++) {
      int q = quants + pair * 32;
      int lowGroup = 2 * pair;
      int highGroup = lowGroup + 1;
      float lowScale = d * qKScale(raw, scales, lowGroup);
      float lowMin = dMin * qKMin(raw, scales, lowGroup);
      float highScale = d * qKScale(raw, scales, highGroup);
      float highMin = dMin * qKMin(raw, scales, highGroup);
      int lowOut = o + lowGroup * 32;
      int highOut = o + highGroup * 32;
      int i = 0;
      for (; i < bound; i += LANES) {
        IntVector packed =
            (IntVector)
                ByteVector.fromArray(LANE_BYTE_SPECIES, raw, q + i)
                    .convertShape(VectorOperators.B2I, LANE_INT_SPECIES, 0);
        ((FloatVector) packed.and(0x0F).convertShape(VectorOperators.I2F, FLOAT_SPECIES, 0))
            .mul(lowScale)
            .sub(lowMin)
            .intoArray(dst, lowOut + i);
        ((FloatVector)
                packed
                    .lanewise(VectorOperators.LSHR, 4)
                    .and(0x0F)
                    .convertShape(VectorOperators.I2F, FLOAT_SPECIES, 0))
            .mul(highScale)
            .sub(highMin)
            .intoArray(dst, highOut + i);
      }
      for (; i < 32; i++) {
        dst[lowOut + i] = lowScale * (raw[q + i] & 0x0F) - lowMin;
        dst[highOut + i] = highScale * ((raw[q + i] >> 4) & 0x0F) - highMin;
      }
    }
  }

  static void q4_KSplit(byte[] raw, int bo, float[] dst, int o, byte[] unpacked) {
    float d = float16(raw, bo);
    float dMin = float16(raw, bo + 2);
    int scales = bo + VectorUtilSupport.GGUF_Q4_K_SCALES_OFFSET;
    int quants = bo + VectorUtilSupport.GGUF_Q4_K_QUANTS_OFFSET;
    int wideBound = WIDE_BYTE_SPECIES.loopBound(32);
    for (int pair = 0; pair < 4; pair++) {
      int q = quants + pair * 32;
      int lowAt = pair * 64;
      int highAt = lowAt + 32;
      int i = 0;
      for (; i < wideBound; i += WIDE_BYTE_SPECIES.length()) {
        ByteVector packed = ByteVector.fromArray(WIDE_BYTE_SPECIES, raw, q + i);
        packed.and((byte) 0x0F).intoArray(unpacked, lowAt + i);
        packed.lanewise(VectorOperators.LSHR, 4).intoArray(unpacked, highAt + i);
      }
      for (; i < 32; i++) {
        unpacked[lowAt + i] = (byte) (raw[q + i] & 0x0F);
        unpacked[highAt + i] = (byte) ((raw[q + i] >> 4) & 0x0F);
      }
    }
    for (int group = 0; group < 8; group++) {
      float scale = d * qKScale(raw, scales, group);
      float min = dMin * qKMin(raw, scales, group);
      int at = group * 32;
      int out = o + at;
      int i = 0;
      if (LANE_MATCHED) {
        int bound = LANE_BYTE_SPECIES.loopBound(32);
        for (; i < bound; i += LANES) {
          ((FloatVector)
                  ByteVector.fromArray(LANE_BYTE_SPECIES, unpacked, at + i)
                      .convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0))
              .mul(scale)
              .sub(min)
              .intoArray(dst, out + i);
        }
      }
      for (; i < 32; i++) {
        dst[out + i] = scale * unpacked[at + i] - min;
      }
    }
  }

  // ---- Q6_K: 128 bytes ql (nibbles), 64 bytes qh (2-bit), 16 int8 scales, d (float16) ----

  static void q6_KByte(byte[] raw, int bo, float[] dst, int o) {
    int qhOffset = bo + VectorUtilSupport.GGUF_Q6_K_QL_BYTES;
    int scales = qhOffset + VectorUtilSupport.GGUF_Q6_K_QH_BYTES;
    float d = float16(raw, scales + VectorUtilSupport.GGUF_Q6_K_SCALES);
    int bound = LANE_BYTE_SPECIES.loopBound(16);
    for (int half = 0; half < 2; half++) {
      int qhBase = qhOffset + half * 32;
      for (int quarter = 0; quarter < 4; quarter++) {
        int qlBase = bo + half * 64 + (quarter & 1) * 32;
        int lowShift = (quarter >> 1) * 4;
        int highShift = 2 * quarter;
        for (int sub = 0; sub < 2; sub++) {
          float scale = d * raw[scales + half * 8 + 2 * quarter + sub];
          int ql = qlBase + sub * 16;
          int qh = qhBase + sub * 16;
          int out = o + half * 128 + quarter * 32 + sub * 16;
          int i = 0;
          for (; i < bound; i += LANES) {
            ByteVector low =
                ByteVector.fromArray(LANE_BYTE_SPECIES, raw, ql + i)
                    .lanewise(VectorOperators.LSHR, lowShift)
                    .and((byte) 0x0F);
            ByteVector high =
                ByteVector.fromArray(LANE_BYTE_SPECIES, raw, qh + i)
                    .lanewise(VectorOperators.LSHR, highShift)
                    .and((byte) 0x03)
                    .lanewise(VectorOperators.LSHL, 4);
            ((FloatVector)
                    low.or(high).sub((byte) 32).convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0))
                .mul(scale)
                .intoArray(dst, out + i);
          }
          for (; i < 16; i++) {
            int quant =
                ((raw[ql + i] >> lowShift) & 0x0F) | (((raw[qh + i] >> highShift) & 0x03) << 4);
            dst[out + i] = scale * (quant - 32);
          }
        }
      }
    }
  }

  static void q6_KInt(byte[] raw, int bo, float[] dst, int o) {
    int qhOffset = bo + VectorUtilSupport.GGUF_Q6_K_QL_BYTES;
    int scales = qhOffset + VectorUtilSupport.GGUF_Q6_K_QH_BYTES;
    float d = float16(raw, scales + VectorUtilSupport.GGUF_Q6_K_SCALES);
    int bound = LANE_BYTE_SPECIES.loopBound(16);
    for (int half = 0; half < 2; half++) {
      int qhBase = qhOffset + half * 32;
      for (int quarter = 0; quarter < 4; quarter++) {
        int qlBase = bo + half * 64 + (quarter & 1) * 32;
        int lowShift = (quarter >> 1) * 4;
        int highShift = 2 * quarter;
        for (int sub = 0; sub < 2; sub++) {
          float scale = d * raw[scales + half * 8 + 2 * quarter + sub];
          int ql = qlBase + sub * 16;
          int qh = qhBase + sub * 16;
          int out = o + half * 128 + quarter * 32 + sub * 16;
          int i = 0;
          for (; i < bound; i += LANES) {
            IntVector low =
                ((IntVector)
                        ByteVector.fromArray(LANE_BYTE_SPECIES, raw, ql + i)
                            .convertShape(VectorOperators.B2I, LANE_INT_SPECIES, 0))
                    .lanewise(VectorOperators.LSHR, lowShift)
                    .and(0x0F);
            IntVector high =
                ((IntVector)
                        ByteVector.fromArray(LANE_BYTE_SPECIES, raw, qh + i)
                            .convertShape(VectorOperators.B2I, LANE_INT_SPECIES, 0))
                    .lanewise(VectorOperators.LSHR, highShift)
                    .and(0x03)
                    .lanewise(VectorOperators.LSHL, 4);
            ((FloatVector) low.or(high).sub(32).convertShape(VectorOperators.I2F, FLOAT_SPECIES, 0))
                .mul(scale)
                .intoArray(dst, out + i);
          }
          for (; i < 16; i++) {
            int quant =
                ((raw[ql + i] >> lowShift) & 0x0F) | (((raw[qh + i] >> highShift) & 0x03) << 4);
            dst[out + i] = scale * (quant - 32);
          }
        }
      }
    }
  }

  static void q6_KSplit(byte[] raw, int bo, float[] dst, int o, byte[] unpacked) {
    int qhOffset = bo + VectorUtilSupport.GGUF_Q6_K_QL_BYTES;
    int scales = qhOffset + VectorUtilSupport.GGUF_Q6_K_QH_BYTES;
    float d = float16(raw, scales + VectorUtilSupport.GGUF_Q6_K_SCALES);
    int wide = WIDE_BYTE_SPECIES.length();
    int wideBound = WIDE_BYTE_SPECIES.loopBound(32);
    for (int half = 0; half < 2; half++) {
      int qhBase = qhOffset + half * 32;
      for (int quarter = 0; quarter < 4; quarter++) {
        int qlBase = bo + half * 64 + (quarter & 1) * 32;
        int lowShift = (quarter >> 1) * 4;
        int highShift = 2 * quarter;
        int at = half * 128 + quarter * 32;
        int i = 0;
        for (; i < wideBound; i += wide) {
          ByteVector low =
              ByteVector.fromArray(WIDE_BYTE_SPECIES, raw, qlBase + i)
                  .lanewise(VectorOperators.LSHR, lowShift)
                  .and((byte) 0x0F);
          ByteVector high =
              ByteVector.fromArray(WIDE_BYTE_SPECIES, raw, qhBase + i)
                  .lanewise(VectorOperators.LSHR, highShift)
                  .and((byte) 0x03)
                  .lanewise(VectorOperators.LSHL, 4);
          low.or(high).sub((byte) 32).intoArray(unpacked, at + i);
        }
        for (; i < 32; i++) {
          int quant =
              ((raw[qlBase + i] >> lowShift) & 0x0F)
                  | (((raw[qhBase + i] >> highShift) & 0x03) << 4);
          unpacked[at + i] = (byte) (quant - 32);
        }
      }
    }
    for (int s = 0; s < 16; s++) {
      float scale = d * raw[scales + s];
      int at = s * 16;
      int out = o + at;
      int i = 0;
      if (LANE_MATCHED) {
        int bound = LANE_BYTE_SPECIES.loopBound(16);
        for (; i < bound; i += LANES) {
          ((FloatVector)
                  ByteVector.fromArray(LANE_BYTE_SPECIES, unpacked, at + i)
                      .convertShape(VectorOperators.B2F, FLOAT_SPECIES, 0))
              .mul(scale)
              .intoArray(dst, out + i);
        }
      }
      for (; i < 16; i++) {
        dst[out + i] = scale * unpacked[at + i];
      }
    }
  }

  private static float float16(byte[] raw, int offset) {
    return Float.float16ToFloat((short) ((raw[offset] & 0xFF) | (raw[offset + 1] << 8)));
  }

  private static int qKScale(byte[] raw, int scales, int group) {
    if (group < 4) {
      return raw[scales + group] & 0x3F;
    }
    return (raw[scales + group + 4] & 0x0F) | (((raw[scales + group - 4] & 0xFF) >>> 6) << 4);
  }

  private static int qKMin(byte[] raw, int scales, int group) {
    if (group < 4) {
      return raw[scales + group + 4] & 0x3F;
    }
    return ((raw[scales + group + 4] & 0xFF) >>> 4) | (((raw[scales + group] & 0xFF) >>> 6) << 4);
  }
}
