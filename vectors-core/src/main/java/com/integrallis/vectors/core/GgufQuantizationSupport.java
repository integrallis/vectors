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

  private static final ThreadLocal<Q6Scratch> Q6_SCRATCH = ThreadLocal.withInitial(Q6Scratch::new);

  private GgufQuantizationSupport() {}

  static void quantizeQ8_0(float[] query, int dimensions, byte[] quants, float[] scales) {
    int blocks = dimensions / VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      int offset = block * VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
      float absoluteMax = 0.0f;
      for (int index = 0; index < VectorUtilSupport.GGUF_Q_BLOCK_SIZE; index++) {
        absoluteMax = Math.max(absoluteMax, Math.abs(query[offset + index]));
      }

      float scale = absoluteMax / 127.0f;
      float inverseScale = absoluteMax == 0.0f ? 0.0f : 127.0f / absoluteMax;
      scales[block] = Float.float16ToFloat(Float.floatToFloat16(scale));
      for (int index = 0; index < VectorUtilSupport.GGUF_Q_BLOCK_SIZE; index++) {
        quants[offset + index] = (byte) ggmlNearestInt(query[offset + index] * inverseScale);
      }
    }
  }

  static void quantizeQ8_K(
      float[] query, int dimensions, byte[] quants, float[] scales, short[] sums) {
    int blockSize = VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE;
    int sumBlockSize = VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE;
    int blocks = dimensions / blockSize;
    for (int block = 0; block < blocks; block++) {
      int offset = block * blockSize;
      float max = 0.0f;
      float absoluteMax = 0.0f;
      for (int index = 0; index < blockSize; index++) {
        float value = query[offset + index];
        float absolute = Math.abs(value);
        if (absolute > absoluteMax) {
          absoluteMax = absolute;
          max = value;
        }
      }

      if (absoluteMax == 0.0f) {
        Arrays.fill(quants, offset, offset + blockSize, (byte) 0);
        if (sums != null) {
          Arrays.fill(sums, offset / sumBlockSize, (offset + blockSize) / sumBlockSize, (short) 0);
        }
        scales[block] = 0.0f;
        continue;
      }

      float inverseScale = -127.0f / max;
      int sum = 0;
      for (int index = 0; index < blockSize; index++) {
        int quant = ggmlNearestInt(inverseScale * query[offset + index]);
        byte stored = (byte) Math.min(127, quant);
        quants[offset + index] = stored;
        if (sums != null) {
          sum += stored;
          if ((index + 1) % sumBlockSize == 0) {
            sums[(offset + index) / sumBlockSize] = (short) sum;
            sum = 0;
          }
        }
      }
      scales[block] = 1.0f / inverseScale;
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

  static Q6Scratch q6Scratch() {
    return Q6_SCRATCH.get();
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }

  static final class Q6Scratch {
    final int[] integerSums = new int[8];
    final float[] laneSums = new float[8];
  }
}
