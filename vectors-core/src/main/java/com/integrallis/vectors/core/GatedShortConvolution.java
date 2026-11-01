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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Shared scalar/SIMD arithmetic contract, with a separate multiply then add at each tap. */
final class GatedShortConvolution {
  private GatedShortConvolution() {}

  static void scalar(
      float[] out,
      int outAt,
      float[] p,
      int pAt,
      int tokens,
      int channels,
      float[] w,
      float[] history,
      int width) {
    apply(out, outAt, p, pAt, tokens, channels, w, history, width, null);
  }

  static void vector(
      float[] out,
      int outAt,
      float[] p,
      int pAt,
      int tokens,
      int channels,
      float[] w,
      float[] history,
      int width,
      VectorSpecies<Float> species) {
    apply(out, outAt, p, pAt, tokens, channels, w, history, width, species);
  }

  private static void apply(
      float[] out,
      int outAt,
      float[] p,
      int pAt,
      int tokens,
      int channels,
      float[] w,
      float[] history,
      int width,
      VectorSpecies<Float> species) {
    int last = (width - 1) * channels;
    for (int token = 0; token < tokens; token++) {
      int base = pAt + token * 3 * channels, dst = outAt + token * channels, i = 0;
      if (species != null)
        for (; i < species.loopBound(channels); i += species.length()) {
          var bx =
              FloatVector.fromArray(species, p, base + i)
                  .mul(FloatVector.fromArray(species, p, base + 2 * channels + i));
          var sum = FloatVector.zero(species);
          for (int tap = 0; tap < width - 1; tap++)
            sum =
                sum.add(
                    FloatVector.fromArray(species, w, tap * channels + i)
                        .mul(FloatVector.fromArray(species, history, tap * channels + i)));
          sum = sum.add(FloatVector.fromArray(species, w, last + i).mul(bx));
          sum.mul(FloatVector.fromArray(species, p, base + channels + i)).intoArray(out, dst + i);
          for (int tap = 0; tap < width - 2; tap++)
            FloatVector.fromArray(species, history, (tap + 1) * channels + i)
                .intoArray(history, tap * channels + i);
          bx.intoArray(history, (width - 2) * channels + i);
        }
      for (; i < channels; i++) {
        float bx = p[base + i] * p[base + 2 * channels + i], sum = 0;
        for (int tap = 0; tap < width - 1; tap++)
          sum += w[tap * channels + i] * history[tap * channels + i];
        sum += w[last + i] * bx;
        out[dst + i] = p[base + channels + i] * sum;
        for (int tap = 0; tap < width - 2; tap++)
          history[tap * channels + i] = history[(tap + 1) * channels + i];
        history[(width - 2) * channels + i] = bx;
      }
    }
  }
}
