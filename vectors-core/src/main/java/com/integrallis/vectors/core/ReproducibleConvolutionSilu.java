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

/**
 * Reproducible SiLU arithmetic within the published float accuracy contract.
 *
 * <p>The ordinary range uses a degree-seven float Taylor polynomial after split-ln2 reduction.
 * Primitive operations avoid interpreted Vector API allocation; Math.fma is used only under the
 * existing configured FMA policy. Large/nonfinite blocks use StrictMath to keep rare lanes stable
 * before and after compilation. Neither route depends on invocation counts or compilation state.
 */
final class ReproducibleConvolutionSilu {
  private ReproducibleConvolutionSilu() {}

  static void apply(float[] values, int offset, int length) {
    int end = offset + length;
    for (int index = offset; index < end; index++) {
      if (!(Math.abs(values[index]) <= 80.0f)) {
        for (int at = offset; at < end; at++) {
          float value = values[at];
          values[at] = value / (1.0f + (float) StrictMath.exp(-value));
        }
        return;
      }
    }
    if (PanamaConstants.HAS_FAST_VECTOR_FMA) {
      for (int at = offset; at < end; at++) {
        float input = values[at], z = -input;
        float biased = z * 1.44269504088896341f + 12582912.0f;
        float n = biased - 12582912.0f;
        float r = Math.fma(n, -0.693359375f, z);
        r = Math.fma(n, 2.12194440e-4f, r);
        float p = Math.fma(1.0f / 5040.0f, r, 1.0f / 720.0f);
        p = Math.fma(p, r, 1.0f / 120.0f);
        p = Math.fma(p, r, 1.0f / 24.0f);
        p = Math.fma(p, r, 1.0f / 6.0f);
        p = Math.fma(p, r, 0.5f);
        p = Math.fma(p, r, 1.0f);
        p = Math.fma(p, r, 1.0f);
        float scale = Float.intBitsToFloat((Float.floatToRawIntBits(biased) << 23) + 0x3f800000);
        values[at] = input / (p * scale + 1.0f);
      }
    } else {
      for (int at = offset; at < end; at++) {
        float input = values[at], z = -input;
        float biased = z * 1.44269504088896341f + 12582912.0f;
        float n = biased - 12582912.0f;
        float r = z - n * 0.693359375f;
        r = r + n * 2.12194440e-4f;
        float p = (1.0f / 5040.0f) * r + 1.0f / 720.0f;
        p = p * r + 1.0f / 120.0f;
        p = p * r + 1.0f / 24.0f;
        p = p * r + 1.0f / 6.0f;
        p = p * r + 0.5f;
        p = p * r + 1.0f;
        p = p * r + 1.0f;
        float scale = Float.intBitsToFloat((Float.floatToRawIntBits(biased) << 23) + 0x3f800000);
        values[at] = input / (p * scale + 1.0f);
      }
    }
  }
}
