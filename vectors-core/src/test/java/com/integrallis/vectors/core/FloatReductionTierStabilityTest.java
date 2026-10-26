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

import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A float reduction must return the same bits in every JIT tier.
 *
 * <p>{@code reduceLanes(ADD)} does not. It leaves the addition order to the compiler, and the
 * interpreted or C1 order differs from C2's, so the first call into a method returns one float and
 * every later call returns another -- one change, at the compilation. A model's first forward pass
 * therefore produced different logits from every later one, which failed the pure-Java determinism
 * audit on HuatuoGPT-o1-7B with trial one diverging and trials two and three agreeing. Whether it
 * was caught depended on when C2 arrived, so the same commit on the same JDK passed on an AMD
 * runner and failed on an Intel one and the defect sat latent in every release.
 *
 * <p>These tests run the reduction far enough past the compilation threshold to cross it.
 */
@Tag("unit")
class FloatReductionTierStabilityTest {

  private static final int ELEMENTS = 4096;

  /** Comfortably past the ~10k invocation C2 threshold for a small method. */
  private static final int ITERATIONS = 100_000;

  private static float[] data(long seed) {
    float[] values = new float[ELEMENTS];
    Random random = new Random(seed);
    for (int index = 0; index < values.length; index++) {
      values[index] = random.nextFloat() - 0.5f;
    }
    return values;
  }

  private static FloatVector accumulate(VectorSpecies<Float> species, float[] a, float[] b) {
    FloatVector accumulator = FloatVector.zero(species);
    for (int index = 0; index < species.loopBound(a.length); index += species.length()) {
      accumulator =
          FloatVector.fromArray(species, a, index)
              .fma(FloatVector.fromArray(species, b, index), accumulator);
    }
    return accumulator;
  }

  private static float fixedOrderDot(VectorSpecies<Float> species, float[] a, float[] b) {
    return PanamaVectorUtilSupport.reduceFloatLanes(accumulate(species, a, b));
  }

  @Test
  void theFixedOrderReductionReturnsTheSameBitsColdAndHot() {
    float[] a = data(42);
    float[] b = data(1337);
    VectorSpecies<Float> species = PanamaVectorUtilSupport.FLOAT_SPECIES;

    int cold = Float.floatToRawIntBits(fixedOrderDot(species, a, b));
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
      assertThat(Float.floatToRawIntBits(fixedOrderDot(species, a, b)))
          .as("fixed-order reduction changed at iteration %d", iteration)
          .isEqualTo(cold);
    }
  }

  /**
   * The guard on the fix itself: it must hold for every float species this class reduces, not only
   * the preferred one, because the rotate shuffles belong to the species they were built for.
   */
  @Test
  void everyFloatSpeciesIsStableAndAgreesWithASequentialSum() {
    float[] a = data(7);
    float[] b = data(9);
    for (VectorSpecies<Float> species :
        java.util.List.of(
            FloatVector.SPECIES_128,
            FloatVector.SPECIES_256,
            PanamaVectorUtilSupport.FLOAT_SPECIES)) {
      int cold = Float.floatToRawIntBits(fixedOrderDot(species, a, b));
      for (int iteration = 0; iteration < 20_000; iteration++) {
        assertThat(Float.floatToRawIntBits(fixedOrderDot(species, a, b)))
            .as("species %s changed at iteration %d", species, iteration)
            .isEqualTo(cold);
      }
      // It is a sum of the same lanes, so it must stay within reduction tolerance of one order.
      float sequential = 0f;
      FloatVector accumulator = accumulate(species, a, b);
      for (int lane = 0; lane < species.length(); lane++) {
        sequential += accumulator.lane(lane);
      }
      assertThat(Float.intBitsToFloat(cold))
          .as("species %s", species)
          .isCloseTo(sequential, org.assertj.core.data.Offset.offset(1e-3f));
    }
  }
}
