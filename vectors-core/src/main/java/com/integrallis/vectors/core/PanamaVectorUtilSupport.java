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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Panama Vector API SIMD implementation of {@link VectorUtilSupport}. Uses FMA-based accumulation
 * with multiple independent accumulators to saturate CPU execution ports.
 *
 * <p>Key patterns (synthesized from Lucene 10.x and JVector):
 *
 * <ul>
 *   <li>4x unrolling for dot product and L2 (1 FMA per iteration)
 *   <li>2x unrolling for cosine (3 FMAs per iteration already saturates ports)
 *   <li>Conditional FMA dispatch via {@link PanamaConstants#HAS_FAST_VECTOR_FMA}
 *   <li>{@code SPECIES_PREFERRED} for main loop with scalar tail
 *   <li>Byte operations use widening conversions (B2S, S2I) to avoid overflow
 * </ul>
 */
final class PanamaVectorUtilSupport implements VectorUtilSupport {

  static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
  static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;
  static final VectorSpecies<Short> SHORT_SPECIES = ShortVector.SPECIES_PREFERRED;
  static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
  static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;
  static final int VECTOR_BITSIZE = FLOAT_SPECIES.vectorBitSize();

  // --- Conditional FMA helpers ---

  static FloatVector fma(FloatVector a, FloatVector b, FloatVector c) {
    if (PanamaConstants.HAS_FAST_VECTOR_FMA) {
      return a.fma(b, c);
    } else {
      return a.mul(b).add(c);
    }
  }

  // --- Float dot product: 4x unrolled FMA (Lucene pattern) ---

  @Override
  public float dotProduct(float[] a, float[] b) {
    return dotProduct(a, 0, b, 0, a.length);
  }

  @Override
  public float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
    int i = 0;
    float res = 0f;

    // Only vectorize if worth the overhead
    if (length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(length);
      res += dotProductBody(a, aOffset, b, bOffset, limit);
      i += limit;
    }

    // Scalar tail
    for (; i < length; i++) {
      res = MathUtil.fma(a[aOffset + i], b[bOffset + i], res);
    }
    return res;
  }

  private float dotProductBody(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();
    int i = 0;

    // Main loop: 4x unrolled for ILP
    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector va1 = FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i);
      FloatVector vb1 = FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i);
      acc1 = fma(va1, vb1, acc1);

      FloatVector va2 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + FLOAT_SPECIES.length());
      FloatVector vb2 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + FLOAT_SPECIES.length());
      acc2 = fma(va2, vb2, acc2);

      FloatVector va3 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 2 * FLOAT_SPECIES.length());
      FloatVector vb3 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + 2 * FLOAT_SPECIES.length());
      acc3 = fma(va3, vb3, acc3);

      FloatVector va4 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 3 * FLOAT_SPECIES.length());
      FloatVector vb4 =
          FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + 3 * FLOAT_SPECIES.length());
      acc4 = fma(va4, vb4, acc4);
    }

    // Vector tail: remaining full vectors
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i);
      acc1 = fma(va, vb, acc1);
    }

    // Reduce 4 accumulators to scalar
    FloatVector res1 = acc1.add(acc2);
    FloatVector res2 = acc3.add(acc4);
    return res1.add(res2).reduceLanes(VectorOperators.ADD);
  }

  // --- Float square distance (L2): 4x unrolled sub+FMA ---

  @Override
  public float squareDistance(float[] a, float[] b) {
    return squareDistance(a, 0, b, 0, a.length);
  }

  @Override
  public float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length) {
    int i = 0;
    float res = 0f;

    if (length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(length);
      res += squareDistanceBody(a, aOffset, b, bOffset, limit);
      i += limit;
    }

    // Scalar tail
    for (; i < length; i++) {
      float diff = a[aOffset + i] - b[bOffset + i];
      res += diff * diff;
    }
    return res;
  }

  private float squareDistanceBody(float[] a, int aOffset, float[] b, int bOffset, int limit) {
    FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();
    int i = 0;

    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector diff1 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i)
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i));
      acc1 = fma(diff1, diff1, acc1);

      FloatVector diff2 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + FLOAT_SPECIES.length())
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i + FLOAT_SPECIES.length()));
      acc2 = fma(diff2, diff2, acc2);

      FloatVector diff3 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 2 * FLOAT_SPECIES.length())
              .sub(
                  FloatVector.fromArray(
                      FLOAT_SPECIES, b, bOffset + i + 2 * FLOAT_SPECIES.length()));
      acc3 = fma(diff3, diff3, acc3);

      FloatVector diff4 =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i + 3 * FLOAT_SPECIES.length())
              .sub(
                  FloatVector.fromArray(
                      FLOAT_SPECIES, b, bOffset + i + 3 * FLOAT_SPECIES.length()));
      acc4 = fma(diff4, diff4, acc4);
    }

    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector diff =
          FloatVector.fromArray(FLOAT_SPECIES, a, aOffset + i)
              .sub(FloatVector.fromArray(FLOAT_SPECIES, b, bOffset + i));
      acc1 = fma(diff, diff, acc1);
    }

    FloatVector res1 = acc1.add(acc2);
    FloatVector res2 = acc3.add(acc4);
    return res1.add(res2).reduceLanes(VectorOperators.ADD);
  }

  // --- Float cosine similarity: 4x unrolled, 3 FMAs per iteration ---

  @Override
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

  // --- Byte dot product: widening to avoid overflow ---
  //
  // Tiered widening strategy (B2I = byte-to-int, 4x lane expansion):
  //   512-bit path: ByteVector.SPECIES_128 (16 bytes) → IntVector.SPECIES_512 (16 ints)
  //   256-bit path: ByteVector.SPECIES_64  ( 8 bytes) → IntVector.SPECIES_256 ( 8 ints)
  //   128-bit path: ByteVector.SPECIES_64  ( 8 bytes) → ShortVector.SPECIES_128 (8 shorts) →
  //                 2× IntVector.SPECIES_128 (lo/hi halves) via S2I; split accumulators
  // Each tier accumulates into its own IntVector and adds to `res` via +=, so tiers compose safely.
  // Int32 overflow is safe: max per-lane accumulation ≤ (dim/lanes) * 127^2 which for dim=1536
  // is ≤ 96 * 16129 ≈ 1.5M, well within int32 range (2.1B).

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int i = 0;
    int res = 0;

    // 512-bit path: 16 bytes → 16 ints per iteration
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        acc = acc.add(ia.mul(ib));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path: 8 bytes → 8 ints per iteration (handles remainder after 512-bit or standalone)
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        acc = acc.add(ia.mul(ib));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path (NEON, RVV-128): 8 bytes → 8 shorts → 2× 4 ints per iteration.
    // Byte*byte products fit in int16 (|a*b| ≤ 127*127 = 16129), so short mul is safe.
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector acc2 = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector prod16 = va16.mul(vb16);
        acc1 =
            acc1.add(
                (IntVector) prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        acc2 =
            acc2.add(
                (IntVector) prod16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
      }
      res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < a.length; i++) {
      res += a[i] * b[i];
    }
    return res;
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    int i = 0;
    int res = 0;

    // 512-bit path
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector diff = ia.sub(ib);
        acc = acc.add(diff.mul(diff));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector diff = ia.sub(ib);
        acc = acc.add(diff.mul(diff));
      }
      res += acc.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path: 8 bytes → 8 shorts → sub in short, then split into 2× 4 ints for mul+add.
    // |a - b| ≤ 255 fits in int16; diff*diff ≤ 65025 does not fit in int16, hence mul in int.
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector acc1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector acc2 = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector diff16 = va16.sub(vb16);
        IntVector diff32_lo =
            (IntVector) diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0);
        IntVector diff32_hi =
            (IntVector) diff16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1);
        acc1 = acc1.add(diff32_lo.mul(diff32_lo));
        acc2 = acc2.add(diff32_hi.mul(diff32_hi));
      }
      res += acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      int diff = a[i] - b[i];
      res += diff * diff;
    }
    return res;
  }

  @Override
  public float cosine(byte[] a, byte[] b) {
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;
    int i = 0;

    // 512-bit path
    if (VECTOR_BITSIZE >= 512 && a.length >= ByteVector.SPECIES_128.length()) {
      int limit = i + ByteVector.SPECIES_128.loopBound(a.length - i);
      IntVector vSum = IntVector.zero(IntVector.SPECIES_512);
      IntVector vNorm1 = IntVector.zero(IntVector.SPECIES_512);
      IntVector vNorm2 = IntVector.zero(IntVector.SPECIES_512);
      for (; i < limit; i += ByteVector.SPECIES_128.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_128, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0);
        vSum = vSum.add(ia.mul(ib));
        vNorm1 = vNorm1.add(ia.mul(ia));
        vNorm2 = vNorm2.add(ib.mul(ib));
      }
      sum += vSum.reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1.reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2.reduceLanes(VectorOperators.ADD);
    }

    // 256-bit path
    if (VECTOR_BITSIZE >= 256 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector vSum = IntVector.zero(IntVector.SPECIES_256);
      IntVector vNorm1 = IntVector.zero(IntVector.SPECIES_256);
      IntVector vNorm2 = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        vSum = vSum.add(ia.mul(ib));
        vNorm1 = vNorm1.add(ia.mul(ia));
        vNorm2 = vNorm2.add(ib.mul(ib));
      }
      sum += vSum.reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1.reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2.reduceLanes(VectorOperators.ADD);
    }

    // 128-bit path: 8 bytes → 8 shorts, multiply in short, widen halves to int for accumulation.
    // All three quantities (a*b, a*a, b*b) stay within int16 range (≤ 16129 for int8 inputs).
    if (VECTOR_BITSIZE == 128 && a.length - i >= ByteVector.SPECIES_64.length()) {
      int limit = i + ByteVector.SPECIES_64.loopBound(a.length - i);
      IntVector vSum1 = IntVector.zero(IntVector.SPECIES_128);
      IntVector vSum2 = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm1Lo = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm1Hi = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm2Lo = IntVector.zero(IntVector.SPECIES_128);
      IntVector vNorm2Hi = IntVector.zero(IntVector.SPECIES_128);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        ShortVector va16 =
            (ShortVector) va.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector vb16 =
            (ShortVector) vb.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
        ShortVector dot16 = va16.mul(vb16);
        ShortVector n1_16 = va16.mul(va16);
        ShortVector n2_16 = vb16.mul(vb16);
        vSum1 =
            vSum1.add(
                (IntVector) dot16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vSum2 =
            vSum2.add(
                (IntVector) dot16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
        vNorm1Lo =
            vNorm1Lo.add(
                (IntVector) n1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vNorm1Hi =
            vNorm1Hi.add(
                (IntVector) n1_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
        vNorm2Lo =
            vNorm2Lo.add(
                (IntVector) n2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0));
        vNorm2Hi =
            vNorm2Hi.add(
                (IntVector) n2_16.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1));
      }
      sum += vSum1.add(vSum2).reduceLanes(VectorOperators.ADD);
      norm1 += vNorm1Lo.add(vNorm1Hi).reduceLanes(VectorOperators.ADD);
      norm2 += vNorm2Lo.add(vNorm2Hi).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      sum += a[i] * b[i];
      norm1 += a[i] * a[i];
      norm2 += b[i] * b[i];
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  // --- MemorySegment distance kernels ---

  @Override
  public float dotProduct(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float res = 0f;

    if (dimensions > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - FLOAT_SPECIES.length();

      for (; i < unrolledLimit; i += 2 * FLOAT_SPECIES.length()) {
        long off1 = (long) i * Float.BYTES;
        long off2 = (long) (i + FLOAT_SPECIES.length()) * Float.BYTES;
        FloatVector va1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off1, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off1, ByteOrder.LITTLE_ENDIAN);
        acc1 = fma(va1, vb1, acc1);
        FloatVector va2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off2, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off2, ByteOrder.LITTLE_ENDIAN);
        acc2 = fma(va2, vb2, acc2);
      }

      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        acc1 = fma(va, vb, acc1);
      }
      res = acc1.add(acc2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      res = MathUtil.fma(ai, bi, res);
    }
    return res;
  }

  @Override
  public float squareDistance(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float res = 0f;

    if (dimensions > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc4 = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();

      // 4x unrolled for ILP (matches array squareDistance body)
      for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
        long off1 = (long) i * Float.BYTES;
        long off2 = (long) (i + FLOAT_SPECIES.length()) * Float.BYTES;
        long off3 = (long) (i + 2 * FLOAT_SPECIES.length()) * Float.BYTES;
        long off4 = (long) (i + 3 * FLOAT_SPECIES.length()) * Float.BYTES;

        FloatVector diff1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off1, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off1, ByteOrder.LITTLE_ENDIAN));
        acc1 = fma(diff1, diff1, acc1);

        FloatVector diff2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off2, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off2, ByteOrder.LITTLE_ENDIAN));
        acc2 = fma(diff2, diff2, acc2);

        FloatVector diff3 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off3, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off3, ByteOrder.LITTLE_ENDIAN));
        acc3 = fma(diff3, diff3, acc3);

        FloatVector diff4 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off4, ByteOrder.LITTLE_ENDIAN)
                .sub(
                    FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off4, ByteOrder.LITTLE_ENDIAN));
        acc4 = fma(diff4, diff4, acc4);
      }

      // Vector tail: remaining full vectors
      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector diff =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN)
                .sub(FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN));
        acc1 = fma(diff, diff, acc1);
      }

      FloatVector r1 = acc1.add(acc2);
      FloatVector r2 = acc3.add(acc4);
      res = r1.add(r2).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float diff = ai - bi;
      res += diff * diff;
    }
    return res;
  }

  // --- MemorySegment cosine: 2x unrolled, 3 FMAs per iteration ---

  @Override
  public float cosine(MemorySegment a, MemorySegment b, int dimensions) {
    int i = 0;
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;

    if (dimensions > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(dimensions);
      FloatVector vSum1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vSum2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vNorm1a = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vNorm1b = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vNorm2a = FloatVector.zero(FLOAT_SPECIES);
      FloatVector vNorm2b = FloatVector.zero(FLOAT_SPECIES);
      int unrolledLimit = limit - FLOAT_SPECIES.length();

      for (; i < unrolledLimit; i += 2 * FLOAT_SPECIES.length()) {
        long off1 = (long) i * Float.BYTES;
        long off2 = (long) (i + FLOAT_SPECIES.length()) * Float.BYTES;

        FloatVector va1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off1, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb1 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off1, ByteOrder.LITTLE_ENDIAN);
        vSum1 = fma(va1, vb1, vSum1);
        vNorm1a = fma(va1, va1, vNorm1a);
        vNorm2a = fma(vb1, vb1, vNorm2a);

        FloatVector va2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off2, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb2 =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off2, ByteOrder.LITTLE_ENDIAN);
        vSum2 = fma(va2, vb2, vSum2);
        vNorm1b = fma(va2, va2, vNorm1b);
        vNorm2b = fma(vb2, vb2, vNorm2b);
      }

      // Vector tail
      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        vSum1 = fma(va, vb, vSum1);
        vNorm1a = fma(va, va, vNorm1a);
        vNorm2a = fma(vb, vb, vNorm2a);
      }

      sum = vSum1.add(vSum2).reduceLanes(VectorOperators.ADD);
      norm1 = vNorm1a.add(vNorm1b).reduceLanes(VectorOperators.ADD);
      norm2 = vNorm2a.add(vNorm2b).reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      sum = MathUtil.fma(ai, bi, sum);
      norm1 = MathUtil.fma(ai, ai, norm1);
      norm2 = MathUtil.fma(bi, bi, norm2);
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  // --- Hamming distance: XOR + BIT_COUNT on LongVector ---

  @Override
  public int hammingDistance(long[] a, long[] b) {
    int i = 0;
    long dist = 0;

    if (a.length >= LONG_SPECIES.length()) {
      int limit = LONG_SPECIES.loopBound(a.length);
      LongVector acc = LongVector.zero(LONG_SPECIES);

      for (; i < limit; i += LONG_SPECIES.length()) {
        LongVector va = LongVector.fromArray(LONG_SPECIES, a, i);
        LongVector vb = LongVector.fromArray(LONG_SPECIES, b, i);
        acc = acc.add(va.lanewise(VectorOperators.XOR, vb).lanewise(VectorOperators.BIT_COUNT));
      }
      dist = acc.reduceLanes(VectorOperators.ADD);
    }

    // Scalar tail
    for (; i < a.length; i++) {
      dist += Long.bitCount(a[i] ^ b[i]);
    }
    return (int) dist;
  }

  // --- Vector arithmetic ---

  @Override
  public void addInPlace(float[] v1, float[] v2) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(v1.length);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, v1, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, v2, i);
      va.add(vb).intoArray(v1, i);
    }
    for (; i < v1.length; i++) {
      v1[i] += v2[i];
    }
  }

  @Override
  public void subInPlace(float[] v1, float[] v2) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(v1.length);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, v1, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, v2, i);
      va.sub(vb).intoArray(v1, i);
    }
    for (; i < v1.length; i++) {
      v1[i] -= v2[i];
    }
  }

  @Override
  public void scale(float[] vector, float multiplier) {
    int i = 0;
    int limit = FLOAT_SPECIES.loopBound(vector.length);
    FloatVector mul = FloatVector.broadcast(FLOAT_SPECIES, multiplier);
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector v = FloatVector.fromArray(FLOAT_SPECIES, vector, i);
      v.mul(mul).intoArray(vector, i);
    }
    for (; i < vector.length; i++) {
      vector[i] *= multiplier;
    }
  }

  @Override
  public float sum(float[] vector) {
    int i = 0;
    float res = 0f;
    int limit = FLOAT_SPECIES.loopBound(vector.length);
    if (limit > 0) {
      FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
      for (; i < limit; i += FLOAT_SPECIES.length()) {
        acc = acc.add(FloatVector.fromArray(FLOAT_SPECIES, vector, i));
      }
      res = acc.reduceLanes(VectorOperators.ADD);
    }
    for (; i < vector.length; i++) {
      res += vector[i];
    }
    return res;
  }

  // --- L2 normalization: 4x unrolled scale ---

  @Override
  public float[] l2normalize(float[] v, boolean throwOnZero) {
    double squaredNorm = dotProduct(v, v);
    if (squaredNorm == 0.0) {
      if (throwOnZero) {
        throw new IllegalArgumentException("Cannot normalize a zero-length vector");
      }
      return v;
    }
    if (MathUtil.isUnitVector((float) squaredNorm)) {
      return v;
    }
    float invNorm = 1.0f / (float) Math.sqrt(squaredNorm);
    int i = 0;

    // Vectorize if large enough
    if (v.length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(v.length);
      l2normalizeBody(v, invNorm, limit);
      i = limit;
    }

    for (; i < v.length; i++) {
      v[i] *= invNorm;
    }
    return v;
  }

  private void l2normalizeBody(float[] v, float invNorm, int limit) {
    FloatVector invNormVector = FloatVector.broadcast(FLOAT_SPECIES, invNorm);
    int i = 0;
    int unrolledLimit = limit - 3 * FLOAT_SPECIES.length();

    // 4x unrolled scale
    for (; i < unrolledLimit; i += 4 * FLOAT_SPECIES.length()) {
      FloatVector.fromArray(FLOAT_SPECIES, v, i).mul(invNormVector).intoArray(v, i);
      FloatVector.fromArray(FLOAT_SPECIES, v, i + FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + FLOAT_SPECIES.length());
      FloatVector.fromArray(FLOAT_SPECIES, v, i + 2 * FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + 2 * FLOAT_SPECIES.length());
      FloatVector.fromArray(FLOAT_SPECIES, v, i + 3 * FLOAT_SPECIES.length())
          .mul(invNormVector)
          .intoArray(v, i + 3 * FLOAT_SPECIES.length());
    }

    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector.fromArray(FLOAT_SPECIES, v, i).mul(invNormVector).intoArray(v, i);
    }
  }

  // --- Fused batch matrix-vector kernels (GEMV) ---

  /**
   * SIMD 4-row-unrolled fused GEMV for dot product.
   *
   * <p>For each group of 4 rows the inner dimension loop loads one SIMD chunk of {@code query}
   * <em>once</em> and multiplies it with the corresponding chunk of all 4 rows simultaneously. This
   * cuts query memory traffic by 4× vs calling {@link #dotProduct} per row and saturates 4 FMA
   * execution ports with independent chains.
   *
   * <p>Rows that do not form a full group of 4 fall back to {@link #dotProduct}.
   */
  @Override
  public void matVecDot(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        acc0 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r0, i), acc0);
        acc1 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r1, i), acc1);
        acc2 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r2, i), acc2);
        acc3 = fma(qv, FloatVector.fromArray(FLOAT_SPECIES, r3, i), acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      // Scalar tail (remaining elements after SIMD loop bound)
      for (int i = limit; i < dim; i++) {
        float q = query[i];
        s0 = MathUtil.fma(q, r0[i], s0);
        s1 = MathUtil.fma(q, r1[i], s1);
        s2 = MathUtil.fma(q, r2[i], s2);
        s3 = MathUtil.fma(q, r3[i], s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    // Tail rows (0–3 remaining)
    for (int r = rowGroup; r < numRows; r++) {
      out[r] = dotProduct(query, 0, matrix[r], 0, dim);
    }
  }

  /**
   * SIMD 4-row-unrolled fused GEMV for squared L2 distance.
   *
   * <p>Same strategy as {@link #matVecDot}: loads each query SIMD chunk once and accumulates
   * squared differences for 4 rows simultaneously.
   */
  @Override
  public void matVecSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3;
    int limit = FLOAT_SPECIES.loopBound(dim);

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      FloatVector acc0 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc1 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc2 = FloatVector.zero(FLOAT_SPECIES);
      FloatVector acc3 = FloatVector.zero(FLOAT_SPECIES);

      for (int i = 0; i < limit; i += FLOAT_SPECIES.length()) {
        FloatVector qv = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        FloatVector d0 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r0, i));
        FloatVector d1 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r1, i));
        FloatVector d2 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r2, i));
        FloatVector d3 = qv.sub(FloatVector.fromArray(FLOAT_SPECIES, r3, i));
        acc0 = fma(d0, d0, acc0);
        acc1 = fma(d1, d1, acc1);
        acc2 = fma(d2, d2, acc2);
        acc3 = fma(d3, d3, acc3);
      }

      float s0 = acc0.reduceLanes(VectorOperators.ADD);
      float s1 = acc1.reduceLanes(VectorOperators.ADD);
      float s2 = acc2.reduceLanes(VectorOperators.ADD);
      float s3 = acc3.reduceLanes(VectorOperators.ADD);

      for (int i = limit; i < dim; i++) {
        float q = query[i];
        float e0 = q - r0[i];
        s0 = MathUtil.fma(e0, e0, s0);
        float e1 = q - r1[i];
        s1 = MathUtil.fma(e1, e1, s1);
        float e2 = q - r2[i];
        s2 = MathUtil.fma(e2, e2, s2);
        float e3 = q - r3[i];
        s3 = MathUtil.fma(e3, e3, s3);
      }

      out[r] = s0;
      out[r + 1] = s1;
      out[r + 2] = s2;
      out[r + 3] = s3;
    }

    for (int r = rowGroup; r < numRows; r++) {
      out[r] = squareDistance(query, 0, matrix[r], 0, dim);
    }
  }
}
