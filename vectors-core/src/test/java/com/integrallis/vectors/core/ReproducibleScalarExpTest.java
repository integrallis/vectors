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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ReproducibleScalarExpTest {
  @Test
  void everyCertifiedRoundingBoundaryAndAdjacentInputMatchesStrictMath() throws Exception {
    int[] catalogue = {
      0xb3000000, 0x33800000, 0x36fdffc1, 0x377eff81, 0x383a3ef1,
      0x38e69cc1, 0x39c6be5b, 0x39e5bb1d, 0xbae0e25c, 0xbbf0edf1,
      0x3d1a274e, 0x4001b249, 0x40315b33, 0xc16912cd, 0x4288942b
    };
    var field = ReproducibleScalarExp.class.getDeclaredField("EXCEPTION_BUCKETS");
    field.setAccessible(true);
    boolean[] buckets = (boolean[]) field.get(null);
    assertThat(buckets).hasSize(2048);
    for (int raw : catalogue) {
      assertThat(buckets[(raw * 0x9e3779b9) >>> 21])
          .as("certified input %08x must take StrictMath even when this JVM agrees", raw)
          .isTrue();
      double reference = StrictMath.exp(Float.intBitsToFloat(raw));
      long bits = Double.doubleToRawLongBits(reference);
      assertThat(Float.floatToRawIntBits((float) Double.longBitsToDouble(bits - 8)))
          .isNotEqualTo(Float.floatToRawIntBits((float) Double.longBitsToDouble(bits + 8)));
      for (int delta = -2; delta <= 2; delta++) exact(Float.intBitsToFloat(raw + delta));
    }
  }

  @Test
  void excludedRangesSignedZeroAndNonfiniteInputsRetainStrictMathBits() {
    int[] edges = {
      0,
      0x80000000,
      1,
      0x80000001,
      0x7f800000,
      0xff800000,
      0x7fc00000,
      0xffc12345,
      0x7f7fffff,
      0xff7fffff,
      0x31800000,
      0xb1800000,
      0x42d00000,
      0xc2d00000,
      0x42b17218,
      0xc2cff1b5
    };
    for (int raw : edges)
      for (int delta = -2; delta <= 2; delta++) exact(Float.intBitsToFloat(raw + delta));
    SplittableRandom random = new SplittableRandom(842066);
    for (int i = 0; i < 100_000; i++) exact(Float.intBitsToFloat(random.nextInt()));
  }

  private static void exact(float value) {
    assertThat(Float.floatToIntBits(ReproducibleScalarExp.exp(value)))
        .as("input bits %08x", Float.floatToRawIntBits(value))
        .isEqualTo(Float.floatToIntBits((float) StrictMath.exp((double) value)));
  }
}
