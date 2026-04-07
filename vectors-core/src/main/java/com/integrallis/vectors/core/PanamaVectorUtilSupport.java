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

  // --- Float cosine similarity: 2x unrolled, 3 FMAs per iteration ---

  @Override
  public float cosine(float[] a, float[] b) {
    int i = 0;
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;

    if (a.length > 2 * FLOAT_SPECIES.length()) {
      int limit = FLOAT_SPECIES.loopBound(a.length);
      float[] result = cosineBody(a, b, limit);
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

  private float[] cosineBody(float[] a, float[] b, int limit) {
    // 2x unrolling (3 FMAs per iteration already adds significant instruction pressure)
    FloatVector sum1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector sum2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector norm1_1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector norm1_2 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector norm2_1 = FloatVector.zero(FLOAT_SPECIES);
    FloatVector norm2_2 = FloatVector.zero(FLOAT_SPECIES);
    int i = 0;
    int unrolledLimit = limit - FLOAT_SPECIES.length();

    for (; i < unrolledLimit; i += 2 * FLOAT_SPECIES.length()) {
      FloatVector va1 = FloatVector.fromArray(FLOAT_SPECIES, a, i);
      FloatVector vb1 = FloatVector.fromArray(FLOAT_SPECIES, b, i);
      sum1 = fma(va1, vb1, sum1);
      norm1_1 = fma(va1, va1, norm1_1);
      norm2_1 = fma(vb1, vb1, norm2_1);

      FloatVector va2 = FloatVector.fromArray(FLOAT_SPECIES, a, i + FLOAT_SPECIES.length());
      FloatVector vb2 = FloatVector.fromArray(FLOAT_SPECIES, b, i + FLOAT_SPECIES.length());
      sum2 = fma(va2, vb2, sum2);
      norm1_2 = fma(va2, va2, norm1_2);
      norm2_2 = fma(vb2, vb2, norm2_2);
    }

    // Vector tail
    for (; i < limit; i += FLOAT_SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
      FloatVector vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
      sum1 = fma(va, vb, sum1);
      norm1_1 = fma(va, va, norm1_1);
      norm2_1 = fma(vb, vb, norm2_1);
    }

    return new float[] {
      sum1.add(sum2).reduceLanes(VectorOperators.ADD),
      norm1_1.add(norm1_2).reduceLanes(VectorOperators.ADD),
      norm2_1.add(norm2_2).reduceLanes(VectorOperators.ADD)
    };
  }

  // --- Byte dot product: widening to avoid overflow ---

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int i = 0;
    int res = 0;

    if (a.length >= 16 && VECTOR_BITSIZE >= 256) {
      int limit = ByteVector.SPECIES_64.loopBound(a.length);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        // Widen bytes to ints, then multiply and accumulate
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        acc = acc.add(ia.mul(ib));
      }
      res = acc.reduceLanes(VectorOperators.ADD);
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

    if (a.length >= 16 && VECTOR_BITSIZE >= 256) {
      int limit = ByteVector.SPECIES_64.loopBound(a.length);
      IntVector acc = IntVector.zero(IntVector.SPECIES_256);
      for (; i < limit; i += ByteVector.SPECIES_64.length()) {
        ByteVector va = ByteVector.fromArray(ByteVector.SPECIES_64, a, i);
        ByteVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, b, i);
        IntVector ia = (IntVector) va.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector ib = (IntVector) vb.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0);
        IntVector diff = ia.sub(ib);
        acc = acc.add(diff.mul(diff));
      }
      res = acc.reduceLanes(VectorOperators.ADD);
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

    if (a.length >= 16 && VECTOR_BITSIZE >= 256) {
      int limit = ByteVector.SPECIES_64.loopBound(a.length);
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
      sum = vSum.reduceLanes(VectorOperators.ADD);
      norm1 = vNorm1.reduceLanes(VectorOperators.ADD);
      norm2 = vNorm2.reduceLanes(VectorOperators.ADD);
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
      FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

      for (; i < limit; i += FLOAT_SPECIES.length()) {
        long off = (long) i * Float.BYTES;
        FloatVector va =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, a, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector vb =
            FloatVector.fromMemorySegment(FLOAT_SPECIES, b, off, ByteOrder.LITTLE_ENDIAN);
        FloatVector diff = va.sub(vb);
        acc = fma(diff, diff, acc);
      }
      res = acc.reduceLanes(VectorOperators.ADD);
    }

    for (; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float diff = ai - bi;
      res += diff * diff;
    }
    return res;
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
}
