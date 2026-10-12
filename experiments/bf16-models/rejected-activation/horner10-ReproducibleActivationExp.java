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
 * Experimental range-reduced double Taylor polynomial with conservative float-midpoint guards.
 * Every ambiguous or exceptional vector is recomputed with StrictMath; see PROTOCOL.md.
 */
final class ReproducibleActivationExp {
  private ReproducibleActivationExp() {}

  private static final int FULL_BITS = PanamaVectorUtilSupport.FLOAT_SPECIES.vectorBitSize();
  private static final VectorSpecies<Double> DOUBLE_SPECIES =
      VectorSpecies.of(double.class, VectorShape.forBitSize(FULL_BITS));
  static final VectorSpecies<Float> FLOAT_SPECIES =
      VectorSpecies.of(float.class, VectorShape.forBitSize(Math.max(64, FULL_BITS / 2)));
  private static final ThreadLocal<double[]> STRICT_SCRATCH =
      ThreadLocal.withInitial(() -> new double[DOUBLE_SPECIES.length()]);

  private static final VectorSpecies<Integer> INT_SPECIES =
      VectorSpecies.of(int.class, FLOAT_SPECIES.vectorShape());
  private static final VectorSpecies<Long> LONG_SPECIES =
      VectorSpecies.of(long.class, DOUBLE_SPECIES.vectorShape());

  private static DoubleVector polynomialExp(DoubleVector x) {
    var half =
        DoubleVector.broadcast(DOUBLE_SPECIES, .5).blend(-.5, x.compare(VectorOperators.LT, 0.0));
    var integers =
        (IntVector)
            x.mul(0x1.71547652b82fep0).add(half).convertShape(VectorOperators.D2I, INT_SPECIES, 0);
    var kd = (DoubleVector) integers.convertShape(VectorOperators.I2D, DOUBLE_SPECIES, 0);
    var k = (LongVector) integers.convertShape(VectorOperators.I2L, LONG_SPECIES, 0);
    var r = x.sub(kd.mul(0x1.62e42feep-1)).sub(kd.mul(0x1.a39ef35793c76p-33));
    var p = DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.27e4fb7789f5cp-22);
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.71de3a556c734p-19));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.a01a01a01a01ap-16));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.a01a01a01a01ap-13));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.6c16c16c16c17p-10));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.1111111111111p-7));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.5555555555555p-5));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.5555555555555p-3));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.0000000000000p-1));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.0000000000000p+0));
    p = p.fma(r, DoubleVector.broadcast(DOUBLE_SPECIES, 0x1.0000000000000p+0));
    return p.reinterpretAsLongs().add(k.lanewise(VectorOperators.LSHL, 52)).reinterpretAsDoubles();
  }

  private static DoubleVector strictExp(DoubleVector input) {
    double[] lanes = STRICT_SCRATCH.get();
    input.intoArray(lanes, 0);
    for (int lane = 0; lane < lanes.length; lane++) lanes[lane] = StrictMath.exp(lanes[lane]);
    return DoubleVector.fromArray(DOUBLE_SPECIES, lanes, 0);
  }

  static FloatVector exp(FloatVector input) {
    if (DOUBLE_SPECIES.length() == 1) {
      return FloatVector.zero(FLOAT_SPECIES)
          .withLane(0, (float) StrictMath.exp(input.lane(0)))
          .withLane(1, (float) StrictMath.exp(input.lane(1)));
    }
    DoubleVector wide = (DoubleVector) input.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, 0);
    DoubleVector result = polynomialExp(wide);
    var low = result.reinterpretAsLongs().and(0x1fffffffL);
    var ambiguous =
        low.compare(VectorOperators.GE, 0xfffc000L)
            .and(low.compare(VectorOperators.LE, 0x10004000L))
            .cast(DOUBLE_SPECIES)
            .or(wide.compare(VectorOperators.LT, -87.0))
            .or(wide.compare(VectorOperators.GT, 88.0))
            .or(wide.test(VectorOperators.IS_FINITE).not());
    if (ambiguous.anyTrue()) result = strictExp(wide);
    return (FloatVector) result.convertShape(VectorOperators.D2F, FLOAT_SPECIES, 0);
  }
}
