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
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/** Shared activation quantization used by scalar and Panama GGUF kernels. */
final class GgufQuantizationSupport {

  private GgufQuantizationSupport() {}

  static void quantizeQ8_0(float[] query, int dimensions, byte[] quants, float[] scales) {
    quantizeQ8_0(query, 0, dimensions, quants, 0, scales, 0);
  }

  static void quantizeQ8_0(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] quants,
      int quantOffset,
      float[] scales,
      int scaleOffset) {
    quantizeQ8_0(
        query, queryOffset, dimensions, quants, quantOffset, scales, scaleOffset, null, 0, false);
  }

  private static void quantizeQ8_0(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] quants,
      int quantOffset,
      float[] scales,
      int scaleOffset,
      int[] zeroPointCorrections,
      int correctionOffset,
      boolean combineCorrectionHalves) {
    int blocks = dimensions / VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
    int correctionsPerBlock = combineCorrectionHalves ? 4 : 8;
    for (int block = 0; block < blocks; block++) {
      int offset = block * VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
      float absoluteMax = 0.0f;
      for (int index = 0; index < VectorUtilSupport.GGUF_Q_BLOCK_SIZE; index++) {
        absoluteMax = Math.max(absoluteMax, Math.abs(query[queryOffset + offset + index]));
      }

      float scale = absoluteMax / 127.0f;
      float inverseScale = absoluteMax == 0.0f ? 0.0f : 127.0f / absoluteMax;
      scales[scaleOffset + block] = Float.float16ToFloat(Float.floatToFloat16(scale));
      int groupSum = 0;
      for (int index = 0; index < VectorUtilSupport.GGUF_Q_BLOCK_SIZE; index++) {
        byte quant = (byte) ggmlNearestInt(query[queryOffset + offset + index] * inverseScale);
        quants[quantOffset + offset + index] = quant;
        if (zeroPointCorrections != null) {
          groupSum += quant;
          if ((index & 3) == 3) {
            int group = index >>> 2;
            int correctionGroup = combineCorrectionHalves ? group & 3 : group;
            int correctionIndex = correctionOffset + block * correctionsPerBlock + correctionGroup;
            int correction = 8 * groupSum;
            if (combineCorrectionHalves && index >= 16) {
              zeroPointCorrections[correctionIndex] += correction;
            } else {
              zeroPointCorrections[correctionIndex] = correction;
            }
            groupSum = 0;
          }
        }
      }
    }
  }

  static void quantizeQ8_0WithQ4Corrections(
      float[] query, int dimensions, byte[] quants, float[] scales, int[] zeroPointCorrections) {
    quantizeQ8_0WithQ4Corrections(
        query, 0, dimensions, quants, 0, scales, 0, zeroPointCorrections, 0);
  }

  static void quantizeQ8_0WithQ4Corrections(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] quants,
      int quantOffset,
      float[] scales,
      int scaleOffset,
      int[] zeroPointCorrections,
      int correctionOffset) {
    quantizeQ8_0(
        query,
        queryOffset,
        dimensions,
        quants,
        quantOffset,
        scales,
        scaleOffset,
        zeroPointCorrections,
        correctionOffset,
        false);
  }

  static void quantizeQ8_0WithCombinedQ4Corrections(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] quants,
      int quantOffset,
      float[] scales,
      int scaleOffset,
      int[] zeroPointCorrections,
      int correctionOffset) {
    quantizeQ8_0(
        query,
        queryOffset,
        dimensions,
        quants,
        quantOffset,
        scales,
        scaleOffset,
        zeroPointCorrections,
        correctionOffset,
        true);
  }

  static void quantizeQ8_K(
      float[] query, int dimensions, byte[] quants, float[] scales, short[] sums) {
    quantizeQ8_K(query, 0, dimensions, quants, 0, scales, 0, sums, 0);
  }

  static void quantizeQ8_K(
      float[] query,
      int queryOffset,
      int dimensions,
      byte[] quants,
      int quantOffset,
      float[] scales,
      int scaleOffset,
      short[] sums,
      int sumOffset) {
    int blockSize = VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE;
    int sumBlockSize = VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE;
    int blocks = dimensions / blockSize;
    for (int block = 0; block < blocks; block++) {
      int offset = block * blockSize;
      float max = 0.0f;
      float absoluteMax = 0.0f;
      for (int index = 0; index < blockSize; index++) {
        float value = query[queryOffset + offset + index];
        float absolute = Math.abs(value);
        if (absolute > absoluteMax) {
          absoluteMax = absolute;
          max = value;
        }
      }

      if (absoluteMax == 0.0f) {
        Arrays.fill(quants, quantOffset + offset, quantOffset + offset + blockSize, (byte) 0);
        if (sums != null) {
          Arrays.fill(
              sums,
              sumOffset + offset / sumBlockSize,
              sumOffset + (offset + blockSize) / sumBlockSize,
              (short) 0);
        }
        scales[scaleOffset + block] = 0.0f;
        continue;
      }

      float inverseScale = -127.0f / max;
      int sum = 0;
      for (int index = 0; index < blockSize; index++) {
        int quant = ggmlNearestInt(inverseScale * query[queryOffset + offset + index]);
        byte stored = (byte) Math.min(127, quant);
        quants[quantOffset + offset + index] = stored;
        if (sums != null) {
          sum += stored;
          if ((index + 1) % sumBlockSize == 0) {
            sums[sumOffset + (offset + index) / sumBlockSize] = (short) sum;
            sum = 0;
          }
        }
      }
      scales[scaleOffset + block] = 1.0f / inverseScale;
    }
  }

  static int qKScale(MemorySegment qWeight, long scalesOffset, int group) {
    if (group < 4) {
      return qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group) & 0x3F;
    }
    int low = qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group + 4L) & 0x0F;
    int high = (qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group - 4L) & 0xFF) >>> 6;
    return low | (high << 4);
  }

  static int qKMin(MemorySegment qWeight, long scalesOffset, int group) {
    if (group < 4) {
      return qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group + 4L) & 0x3F;
    }
    int low = (qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group + 4L) & 0xFF) >>> 4;
    int high = (qWeight.get(ValueLayout.JAVA_BYTE, scalesOffset + group) & 0xFF) >>> 6;
    return low | (high << 4);
  }

  /**
   * Unpacks all 8 scales and 8 mins from a Q4_K block's scale bytes in one pass.
   *
   * <p>This replaces 16 scalar per-group calls to {@link #qKScale} / {@link #qKMin}
   * with a single batch extraction, matching the vectorized unpack in the native
   * kernel (commit dcd8b172). Layout is ggml's, from
   * {@code ggml_vec_dot_q4_K_q8_K} (MIT, ggerganov/ggml).
   *
   * @param qWeight the weight memory segment
   * @param scalesOffset offset to the 12 bytes of scale/min data within the block
   * @param scalesOut pre-allocated int[8] to receive scales[0..7]
   * @param minsOut pre-allocated int[8] to receive mins[0..7]
   */
  static void qKUnpackScalesAndMins(
      MemorySegment qWeight,
      long scalesOffset,
      int[] scalesOut,
      int[] minsOut) {
    // Read 12 bytes as three little-endian u32 words.
    int s0 = qWeight.get(ValueLayout.JAVA_INT_UNALIGNED, scalesOffset);
    int s1 = qWeight.get(ValueLayout.JAVA_INT_UNALIGNED, scalesOffset + 4);
    int s2 = qWeight.get(ValueLayout.JAVA_INT_UNALIGNED, scalesOffset + 8);

    // scales[0..3] = s0 & 0x3F3F3F3F (low 6 bits of each byte)
    scalesOut[0] = s0 & 0x3F;
    scalesOut[1] = (s0 >>> 8) & 0x3F;
    scalesOut[2] = (s0 >>> 16) & 0x3F;
    scalesOut[3] = (s0 >>> 24) & 0x3F;

    // scales[4..7] = (s2 & 0x0F0F0F0F) | (((s0 >>> 6) & 0x03030303) << 4)
    int scales47 = (s2 & 0x0F0F0F0F) | (((s0 >>> 6) & 0x03030303) << 4);
    scalesOut[4] = scales47 & 0x3F;
    scalesOut[5] = (scales47 >>> 8) & 0x3F;
    scalesOut[6] = (scales47 >>> 16) & 0x3F;
    scalesOut[7] = (scales47 >>> 24) & 0x3F;

    // mins[0..3] = s1 & 0x3F3F3F3F
    minsOut[0] = s1 & 0x3F;
    minsOut[1] = (s1 >>> 8) & 0x3F;
    minsOut[2] = (s1 >>> 16) & 0x3F;
    minsOut[3] = (s1 >>> 24) & 0x3F;

    // mins[4..7] = ((s2 >>> 4) & 0x0F0F0F0F) | (((s1 >>> 6) & 0x03030303) << 4)
    int mins47 = ((s2 >>> 4) & 0x0F0F0F0F) | (((s1 >>> 6) & 0x03030303) << 4);
    minsOut[4] = mins47 & 0x3F;
    minsOut[5] = (mins47 >>> 8) & 0x3F;
    minsOut[6] = (mins47 >>> 16) & 0x3F;
    minsOut[7] = (mins47 >>> 24) & 0x3F;
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }
}
