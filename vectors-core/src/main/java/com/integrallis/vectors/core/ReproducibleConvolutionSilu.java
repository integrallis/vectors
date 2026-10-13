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
 * Stable SiLU within the published absolute float accuracy contract.
 *
 * <p>A degree-seven Taylor polynomial approximates exp(-x/16); four squarings recover exp(-x). The
 * fixed ordinary interval is [-32, 32]. Outside it, the entire block uses StrictMath so mixed
 * ordinary and exceptional inputs retain the same result before and after compilation. Primitive
 * array loops permit SIMD compilation without temporary vector objects or retained scratch arrays.
 *
 * <p>Published support tests permit 2e-6 absolute SiLU error; this is not a small-ULP bound:
 * relative error grows for the tiny negative outputs near -32. Arithmetic order and the existing
 * configured FMA policy are fixed; invocation counts and compilation state never select a route.
 */
final class ReproducibleConvolutionSilu {
  private ReproducibleConvolutionSilu() {}

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
        float p = Math.fma(1f / 5040f, r, 1f / 720f);
        p = Math.fma(p, r, 1f / 120f);
        p = Math.fma(p, r, 1f / 24f);
        p = Math.fma(p, r, 1f / 6f);
        p = Math.fma(p, r, .5f);
        p = Math.fma(p, r, 1f);
        p = Math.fma(p, r, 1f);
        p = p * p;
        p = p * p;
        p = p * p;
        p = p * p;
        values[i] = x / (1f + p);
      }
    } else {
      for (int i = offset; i < end; i++) {
        float x = values[i], r = -x * 0.0625f;
        float p = (1f / 5040f) * r + 1f / 720f;
        p = p * r + 1f / 120f;
        p = p * r + 1f / 24f;
        p = p * r + 1f / 6f;
        p = p * r + .5f;
        p = p * r + 1f;
        p = p * r + 1f;
        p = p * p;
        p = p * p;
        p = p * p;
        p = p * p;
        values[i] = x / (1f + p);
      }
    }
  }
}
