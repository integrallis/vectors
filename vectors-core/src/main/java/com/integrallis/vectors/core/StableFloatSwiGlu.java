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
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Stable float SwiGLU based on Arm optimized-routines AdvSIMD expf. Copyright(c)2019-2026 Arm
 * Limited. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception. Fixed Arm coefficients;
 * explicit fused arithmetic. Nearest-even range reduction replaces Arm's nearest-away ties; both
 * retain the same half-ln2 reduced interval. No task-data tuning.
 */
final class StableFloatSwiGlu {
  private static final VectorSpecies<Float> S = PanamaVectorUtilSupport.FLOAT_SPECIES;
  private static final VectorSpecies<Integer> IS = S.withLanes(int.class);
  private static final FloatVector INV = FloatVector.broadcast(S, 0x1.715476p0f);
  private static final FloatVector SHIFT = FloatVector.broadcast(S, 0x1.8p23f);
  private static final FloatVector HI = FloatVector.broadcast(S, 0x1.62e4p-1f);
  private static final FloatVector LO = FloatVector.broadcast(S, 0x1.7f7d1cp-20f);
  private static final FloatVector C0 = FloatVector.broadcast(S, 0x1.0e4020p-7f);
  private static final FloatVector C1 = FloatVector.broadcast(S, 0x1.573e2ep-5f);
  private static final FloatVector C2 = FloatVector.broadcast(S, 0x1.555e66p-3f);
  private static final FloatVector C3 = FloatVector.broadcast(S, 0x1.fffdb6p-2f);
  private static final FloatVector C4 = FloatVector.broadcast(S, 0x1.ffffecp-1f);
  private static final FloatVector ONE = FloatVector.broadcast(S, 1f);
  private static final FloatVector BOUND = FloatVector.broadcast(S, 80f);
  private static final IntVector BIAS = IntVector.broadcast(IS, 0x3f800000);

  static void apply(
      float[] out, int outAt, float[] gate, int gateAt, float[] up, int upAt, int length) {
    int i = 0, limit = S.loopBound(length);
    for (; i < limit; i += S.length()) {
      var value = FloatVector.fromArray(S, gate, gateAt + i);
      var multiplier = FloatVector.fromArray(S, up, upAt + i);
      var x = value.neg();
      if (x.abs().compare(VectorOperators.LE, BOUND).allTrue()) {
        var n = x.mul(INV).add(SHIFT).sub(SHIFT);
        var r = n.neg().fma(HI, x);
        r = n.neg().fma(LO, r);
        var k = (IntVector) n.convertShape(VectorOperators.F2I, IS, 0);
        var scale = k.lanewise(VectorOperators.LSHL, 23).add(BIAS).reinterpretAsFloats();
        var r2 = r.mul(r);
        var p = r.fma(C0, C1);
        var q = r.fma(C2, C3);
        q = p.fma(r2, q);
        p = C4.mul(r);
        var poly = q.fma(r2, p);
        var e = scale.fma(poly, scale);
        value.div(ONE.add(e)).mul(multiplier).intoArray(out, outAt + i);
      } else {
        // This block contains an exceptional value. Ordinary lanes retain identical arithmetic.
        for (int lane = 0; lane < S.length(); lane++) {
          float v = gate[gateAt + i + lane];
          float u = up[upAt + i + lane];
          out[outAt + i + lane] = v / (1f + exp(-v)) * u;
        }
      }
    }
    for (; i < length; i++) {
      float v = gate[gateAt + i];
      out[outAt + i] = v / (1f + exp(-v)) * up[upAt + i];
    }
  }

  private static float exp(float x) {
    if (!(Math.abs(x) <= 80f)) return (float) StrictMath.exp(x);
    float n = (x * 0x1.715476p0f + 0x1.8p23f) - 0x1.8p23f;
    float r = Math.fma(-n, 0x1.62e4p-1f, x);
    r = Math.fma(-n, 0x1.7f7d1cp-20f, r);
    float scale = Float.intBitsToFloat(((int) n << 23) + 0x3f800000);
    float r2 = r * r;
    float p = Math.fma(r, 0x1.0e4020p-7f, 0x1.573e2ep-5f);
    float q = Math.fma(r, 0x1.555e66p-3f, 0x1.fffdb6p-2f);
    q = Math.fma(p, r2, q);
    p = 0x1.ffffecp-1f * r;
    float poly = Math.fma(q, r2, p);
    return Math.fma(scale, poly, scale);
  }
}
