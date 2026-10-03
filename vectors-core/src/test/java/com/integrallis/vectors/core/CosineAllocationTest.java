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

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

import java.lang.management.ManagementFactory;
import java.util.SplittableRandom;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CosineAllocationTest {
  private static volatile float sink;

  @Test
  void warmedCosineDoesNotAllocateATemporaryArrayPerPair() {
    var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    assumeTrue(bean.isThreadAllocatedMemorySupported());
    bean.setThreadAllocatedMemoryEnabled(true);
    var implementation = new PanamaVectorUtilSupport();
    long thread = Thread.currentThread().threadId();
    for (int dimension : new int[] {PanamaVectorUtilSupport.FLOAT_SPECIES.length(), 512}) {
      float[] a = new float[dimension], b = new float[dimension];
      java.util.Arrays.fill(a, .25f);
      java.util.Arrays.fill(b, .5f);
      for (int i = 0; i < 100000; i++) {
        a[0] = (i % 97) * .001f;
        sink = implementation.cosine(a, b);
      }
      long before = bean.getThreadAllocatedBytes(thread);
      for (int i = 0; i < 20000; i++) {
        a[0] = (i % 97) * .001f;
        sink = implementation.cosine(a, b);
      }
      long allocated = bean.getThreadAllocatedBytes(thread) - before;
      System.out.printf(
          "COSINE_ALLOC dimension=%d iterations=20000 bytes=%d%n", dimension, allocated);
      assertTrue(
          allocated < 20000L * 8,
          "cosine allocated " + allocated + " bytes for dimension " + dimension);
      assertTrue(Float.isFinite(sink));
    }
  }

  @Test
  void cosinePreservesStartingRevisionBitsAtVectorBoundariesAndTails() {
    var implementation = new PanamaVectorUtilSupport();
    var reference = new LegacyCosine();
    SplittableRandom random = new SplittableRandom(83421);
    for (int dimension :
        new int[] {
          1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 257, 511, 512,
          513, 768, 960, 961
        }) {
      for (int trial = 0; trial < 100; trial++) {
        float[] a = new float[dimension], b = new float[dimension];
        for (int d = 0; d < dimension; d++) {
          a[d] = Math.scalb((float) random.nextDouble(-1, 1), random.nextInt(-10, 11));
          b[d] = Math.scalb((float) random.nextDouble(-1, 1), random.nextInt(-10, 11));
        }
        if (trial == 0) java.util.Arrays.fill(a, 0f);
        assertEquals(
            Float.floatToIntBits(reference.cosine(a, b)),
            Float.floatToIntBits(implementation.cosine(a, b)),
            "dimension=" + dimension);
      }
    }
  }

  // Arithmetic oracle copied from 4bc505b, before removal of the temporary result arrays.
  // Comparing raw finite values catches changes to FMA grouping, reductions and scalar tails.
  private static final class LegacyCosine {
    private static final VectorSpecies<Float> FLOAT_SPECIES = PanamaVectorUtilSupport.FLOAT_SPECIES;

    private static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
      return PanamaVectorUtilSupport.fma(a, b, c);
    }

    public float cosine(float[] a, float[] b) {
      int i = 0;
      float sum = 0f;
      float norm1 = 0f;
      float norm2 = 0f;

      if (a.length >= 4 * FLOAT_SPECIES.length()) {
        // 4x unrolled main body: 12 independent FMA accumulators hide FMA latency on NEON/AVX.
        int limit = FLOAT_SPECIES.loopBound(a.length);
        float[] result = cosineBody4x(a, b, limit);
        sum = result[0];
        norm1 = result[1];
        norm2 = result[2];
        i = limit;
      } else if (a.length >= FLOAT_SPECIES.length()) {
        // Short vectors: single-accumulator vector body (no unroll), avoids unroll-prologue cost.
        int limit = FLOAT_SPECIES.loopBound(a.length);
        float[] result = cosineBody1x(a, b, limit);
        sum = result[0];
        norm1 = result[1];
        norm2 = result[2];
        i = limit;
      }

      // Scalar tail
      for (; i < a.length; i++) {
        sum = MathUtil.fma(a[i], b[i], sum);
        norm1 = MathUtil.fma(a[i], a[i], norm1);
        norm2 = MathUtil.fma(b[i], b[i], norm2);
      }

      return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
    }

    private float[] cosineBody4x(float[] a, float[] b, int limit) {
      FloatVector s0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector s3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1_3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2_3 = FloatVector.zero(FLOAT_SPECIES);
      int i = 0;
      int lanes = FLOAT_SPECIES.length();
      int unrolledLimit = limit - 3 * lanes;

      for (; i < unrolledLimit; i += 4 * lanes) {
        FloatVector va0 = FloatVector.fromArray(FLOAT_SPECIES, a, i);
        FloatVector vb0 = FloatVector.fromArray(FLOAT_SPECIES, b, i);
        FloatVector va1 = FloatVector.fromArray(FLOAT_SPECIES, a, i + lanes);
        FloatVector vb1 = FloatVector.fromArray(FLOAT_SPECIES, b, i + lanes);
        FloatVector va2 = FloatVector.fromArray(FLOAT_SPECIES, a, i + 2 * lanes);
        FloatVector vb2 = FloatVector.fromArray(FLOAT_SPECIES, b, i + 2 * lanes);
        FloatVector va3 = FloatVector.fromArray(FLOAT_SPECIES, a, i + 3 * lanes);
        FloatVector vb3 = FloatVector.fromArray(FLOAT_SPECIES, b, i + 3 * lanes);

        s0 = fma(va0, vb0, s0);
        s1 = fma(va1, vb1, s1);
        s2 = fma(va2, vb2, s2);
        s3 = fma(va3, vb3, s3);
        n1_0 = fma(va0, va0, n1_0);
        n1_1 = fma(va1, va1, n1_1);
        n1_2 = fma(va2, va2, n1_2);
        n1_3 = fma(va3, va3, n1_3);
        n2_0 = fma(vb0, vb0, n2_0);
        n2_1 = fma(vb1, vb1, n2_1);
        n2_2 = fma(vb2, vb2, n2_2);
        n2_3 = fma(vb3, vb3, n2_3);
      }

      // Vector tail (1 lane-width at a time)
      for (; i < limit; i += lanes) {
        FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
        s0 = fma(va, vb, s0);
        n1_0 = fma(va, va, n1_0);
        n2_0 = fma(vb, vb, n2_0);
      }

      return new float[] {
        s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD),
        n1_0.add(n1_1).add(n1_2.add(n1_3)).reduceLanes(VectorOperators.ADD),
        n2_0.add(n2_1).add(n2_2.add(n2_3)).reduceLanes(VectorOperators.ADD)
      };
    }

    private float[] cosineBody1x(float[] a, float[] b, int limit) {
      FloatVector s = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector n2 = FloatVector.zero(FLOAT_SPECIES);
      int lanes = FLOAT_SPECIES.length();
      for (int i = 0; i < limit; i += lanes) {
        FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
        s = fma(va, vb, s);
        n1 = fma(va, va, n1);
        n2 = fma(vb, vb, n2);
      }
      return new float[] {
        s.reduceLanes(VectorOperators.ADD),
        n1.reduceLanes(VectorOperators.ADD),
        n2.reduceLanes(VectorOperators.ADD)
      };
    }
  }
}
