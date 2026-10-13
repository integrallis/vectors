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
 * Stable SiLU meeting the published support tests’ absolute-error tolerance.
 *
 * <p>A degree-five Chebyshev-series polynomial approximates exp(-x/16); four squarings recover
 * exp(-x). The fixed ordinary interval is [-32, 32]. Outside it, the entire block uses StrictMath
 * so mixed ordinary and exceptional inputs retain the same result before and after compilation.
 * Primitive array loops permit SIMD compilation without temporary vector objects or retained
 * scratch arrays.
 *
 * <p>Published support tests permit 2e-6 absolute SiLU error; this is not a small-ULP bound:
 * relative error grows for the tiny negative outputs near -32. Arithmetic order and the existing
 * configured FMA policy are fixed; invocation counts and compilation state never select a route.
 */
final class ReproducibleConvolutionSilu {
  private ReproducibleConvolutionSilu() {}

  // Coefficients: truncate exp(a*t) = I0(a) + 2*sum(Ik(a)*Tk(t)) at degree five,
  // with a=3/8 and t=r/a. Ik(a)=sum_m (a/2)^(2m+k)/(m!*(m+k)!). Evaluate 40 terms
  // at 80 decimal digits, convert Chebyshev to monomial form, then round once to float.
  // This central fit favors the sensitive SiLU region; the full [-32,32] domain is checked
  // separately against the absolute-error contract. Hex literals preserve the rounded values.

  static void apply(float[] values, int offset, int length) {
    int end = offset + length;
    float maximum = 0f;
    for (int i = offset; i < end; i++) maximum = Math.max(maximum, Math.abs(values[i]));
    if (!(maximum <= 32f)) {
      for (int i = offset; i < end; i++) {
        float x = values[i];
        values[i] = x / (1f + (float) StrictMath.exp(-x));
      }
      return;
    }
    if (PanamaConstants.HAS_FAST_VECTOR_FMA) {
      for (int i = offset; i < end; i++) {
        float x = values[i], r = -x * 0.0625f;
        float p = Math.fma(0x1.12abb20000000p-7f, r, 0x1.57bd8a0000000p-5f);
        p = Math.fma(p, r, 0x1.5553860000000p-3f);
        p = Math.fma(p, r, 0x1.fffbf00000000p-2f);
        p = Math.fma(p, r, 0x1.0000020000000p+0f);
        p = Math.fma(p, r, 0x1.0000020000000p+0f);
        p = p * p;
        p = p * p;
        p = p * p;
        p = p * p;
        values[i] = x / (1f + p);
      }
    } else {
      for (int i = offset; i < end; i++) {
        float x = values[i], r = -x * 0.0625f;
        float p = 0x1.12abb20000000p-7f * r + 0x1.57bd8a0000000p-5f;
        p = p * r + 0x1.5553860000000p-3f;
        p = p * r + 0x1.fffbf00000000p-2f;
        p = p * r + 0x1.0000020000000p+0f;
        p = p * r + 0x1.0000020000000p+0f;
        p = p * p;
        p = p * p;
        p = p * p;
        p = p * p;
        values[i] = x / (1f + p);
      }
    }
  }
}
