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
package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.VectorizationProvider;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD population-count helpers over packed {@code long[]} bit arrays. Both operations dispatch to
 * a {@link LongVector} {@code BIT_COUNT} kernel when the Panama SIMD provider is active and fall
 * back to a scalar {@link Long#bitCount} loop otherwise — the same primitive {@code
 * PanamaVectorUtilSupport.hammingDistance} uses (XOR + {@code BIT_COUNT}), reused here for AND
 * (asymmetric bit-plane dot) and plain popcount.
 *
 * <p>Package-private — an internal quantization-scoring detail. The {@link LongVector} references
 * are confined to the nested {@code Simd} class so its symbols only link when the Panama provider
 * is present; the {@code SIMD} gate plus a defensive {@link LinkageError} catch keep this class
 * usable on a runtime without {@code jdk.incubator.vector}.
 */
final class BitCounts {

  /** True when the SIMD (Panama) provider is active; gates the {@link LongVector} fast paths. */
  private static final boolean SIMD = VectorizationProvider.isPanamaEnabled();

  private BitCounts() {}

  /** Returns the total number of 1-bits across all longs in {@code bits}. */
  static int popcount(long[] bits) {
    if (SIMD) {
      try {
        return Simd.popcount(bits);
      } catch (LinkageError ignored) {
        // jdk.incubator.vector unavailable at runtime — fall through to the scalar path.
      }
    }
    return scalarPopcount(bits);
  }

  /**
   * Returns {@code sum_i Long.bitCount(a[i] & b[i])} — the number of positions where both bit
   * arrays have a 1-bit. Requires {@code a.length == b.length}.
   */
  static int popcountAnd(long[] a, long[] b) {
    if (SIMD) {
      try {
        return Simd.popcountAnd(a, b);
      } catch (LinkageError ignored) {
        // jdk.incubator.vector unavailable at runtime — fall through to the scalar path.
      }
    }
    return scalarPopcountAnd(a, b);
  }

  static int scalarPopcount(long[] bits) {
    int count = 0;
    for (long l : bits) {
      count += Long.bitCount(l);
    }
    return count;
  }

  static int scalarPopcountAnd(long[] a, long[] b) {
    int count = 0;
    for (int i = 0; i < a.length; i++) {
      count += Long.bitCount(a[i] & b[i]);
    }
    return count;
  }

  /** Isolated so {@link LongVector} symbols only resolve when the Panama provider is present. */
  private static final class Simd {

    private static final VectorSpecies<Long> SPECIES = LongVector.SPECIES_PREFERRED;

    private Simd() {}

    static int popcount(long[] bits) {
      int i = 0;
      long count = 0;
      int len = bits.length;
      if (len >= SPECIES.length()) {
        int limit = SPECIES.loopBound(len);
        LongVector acc = LongVector.zero(SPECIES);
        for (; i < limit; i += SPECIES.length()) {
          acc = acc.add(LongVector.fromArray(SPECIES, bits, i).lanewise(VectorOperators.BIT_COUNT));
        }
        count = acc.reduceLanes(VectorOperators.ADD);
      }
      for (; i < len; i++) {
        count += Long.bitCount(bits[i]);
      }
      return (int) count;
    }

    static int popcountAnd(long[] a, long[] b) {
      int i = 0;
      long count = 0;
      int len = a.length;
      if (len >= SPECIES.length()) {
        int limit = SPECIES.loopBound(len);
        LongVector acc = LongVector.zero(SPECIES);
        for (; i < limit; i += SPECIES.length()) {
          LongVector va = LongVector.fromArray(SPECIES, a, i);
          LongVector vb = LongVector.fromArray(SPECIES, b, i);
          acc = acc.add(va.lanewise(VectorOperators.AND, vb).lanewise(VectorOperators.BIT_COUNT));
        }
        count = acc.reduceLanes(VectorOperators.ADD);
      }
      for (; i < len; i++) {
        count += Long.bitCount(a[i] & b[i]);
      }
      return (int) count;
    }
  }
}
