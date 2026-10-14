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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Activation arithmetic that stays stable across JVM compilation modes.
 *
 * <p>Ordinary SwiGLU uses the fixed float polynomial in {@link StableFloatSwiGlu}. Forward
 * overlapping output aliases retain the original vector-group read/write order and its unfused
 * double-vector exponential; this overlap path reproduces the float cast of StrictMath.exp.
 *
 * <p>The retained overlap exponential is adapted from fdlibm e_exp.c,
 * https://netlib.org/fdlibm/e_exp.c. Copyright (C) 2004 by Sun Microsystems, Inc. All rights
 * reserved. Permission to use, copy, modify, and distribute this software is freely granted,
 * provided that this notice is preserved.
 */
final class ReproducibleActivationExp {
  private ReproducibleActivationExp() {}

  private static final int FULL_BITS = PanamaVectorUtilSupport.FLOAT_SPECIES.vectorBitSize();
  private static final VectorSpecies<Double> DOUBLE_SPECIES =
      VectorSpecies.of(double.class, VectorShape.forBitSize(FULL_BITS));
  static final VectorSpecies<Float> FLOAT_SPECIES =
      VectorSpecies.of(float.class, VectorShape.forBitSize(Math.max(64, FULL_BITS / 2)));
  private static final VectorSpecies<Integer> INT_SPECIES =
      VectorSpecies.of(int.class, FLOAT_SPECIES.vectorShape());
  private static final VectorSpecies<Long> LONG_SPECIES =
      VectorSpecies.of(long.class, DOUBLE_SPECIES.vectorShape());
  private static final double HI = 0x1.62e42feep-1, LO = 0x1.a39ef35793c76p-33;

  /** Keeps the full vector computation behind an array-only call boundary. */
  static void swiGlu(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int length) {
    if ((out == gate && outOffset > gateOffset && outOffset < gateOffset + length)
        || (out == up && outOffset > upOffset && outOffset < upOffset + length)) {
      swiGluAliased(out, outOffset, gate, gateOffset, up, upOffset, length);
    } else StableFloatSwiGlu.apply(out, outOffset, gate, gateOffset, up, upOffset, length);
  }

  private static void swiGluAliased(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int length) {
    int index = 0;
    int limit = PanamaVectorUtilSupport.FLOAT_SPECIES.loopBound(length);
    FloatVector oneFloat = FloatVector.broadcast(FLOAT_SPECIES, 1.0f);
    for (; index < limit; index += PanamaVectorUtilSupport.FLOAT_SPECIES.length()) {
      // Capture the complete published read group before writing any overlapping output.
      FloatVector gateGroup =
          FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, gate, gateOffset + index);
      FloatVector upGroup =
          FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, up, upOffset + index);
      for (int part = 0;
          part < PanamaVectorUtilSupport.FLOAT_SPECIES.length() / FLOAT_SPECIES.length();
          part++) {
        FloatVector value = (FloatVector) gateGroup.reinterpretShape(FLOAT_SPECIES, part);
        FloatVector multiplier = (FloatVector) upGroup.reinterpretShape(FLOAT_SPECIES, part);
        FloatVector input = value.neg();
        FloatVector result;
        if (DOUBLE_SPECIES.length() == 1) {
          result =
              FloatVector.zero(FLOAT_SPECIES)
                  .withLane(0, (float) StrictMath.exp(input.lane(0)))
                  .withLane(1, (float) StrictMath.exp(input.lane(1)));
        } else {
          var bounded = input.max(-104.0f).min(89.0f);
          DoubleVector x =
              (DoubleVector) bounded.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, 0);
          var half =
              DoubleVector.broadcast(DOUBLE_SPECIES, .5)
                  .blend(-.5, x.compare(VectorOperators.LT, 0.0));
          var integers =
              (IntVector)
                  x.mul(0x1.71547652b82fep0)
                      .add(half)
                      .convertShape(VectorOperators.D2I, INT_SPECIES, 0);
          var kd = (DoubleVector) integers.convertShape(VectorOperators.I2D, DOUBLE_SPECIES, 0);
          var k = (LongVector) integers.convertShape(VectorOperators.I2L, LONG_SPECIES, 0);
          var hi = x.sub(kd.mul(HI));
          var lo = kd.mul(LO);
          var r = hi.sub(lo);
          var t = r.mul(r);
          var polynomial =
              t.mul(0x1.6376972bea4d0p-25)
                  .add(-0x1.bbd41c5d26bf1p-20)
                  .mul(t)
                  .add(0x1.1566aaf25de2cp-14)
                  .mul(t)
                  .add(-0x1.6c16c16bebd93p-9)
                  .mul(t)
                  .add(0x1.555555555553ep-3);
          var c = r.sub(t.mul(polynomial));
          var one = DoubleVector.broadcast(DOUBLE_SPECIES, 1.0);
          var y = one.sub(lo.sub(r.mul(c).div(c.neg().add(2.0))).sub(hi));
          y =
              y.reinterpretAsLongs()
                  .add(k.lanewise(VectorOperators.LSHL, 52))
                  .reinterpretAsDoubles();

          result = (FloatVector) y.convertShape(VectorOperators.D2F, FLOAT_SPECIES, 0);
          result = result.blend(Float.POSITIVE_INFINITY, input.compare(VectorOperators.GT, 89.0f));
          result = result.blend(0.0f, input.compare(VectorOperators.LT, -104.0f));
          result = result.blend(input.add(input), input.test(VectorOperators.IS_NAN));
        }
        value
            .div(oneFloat.add(result))
            .mul(multiplier)
            .intoArray(out, outOffset + index + part * FLOAT_SPECIES.length());
      }
    }
    for (; index < length; index++) {
      float value = gate[gateOffset + index];
      out[outOffset + index] =
          value / (1.0f + (float) StrictMath.exp(-value)) * up[upOffset + index];
    }
  }
}
