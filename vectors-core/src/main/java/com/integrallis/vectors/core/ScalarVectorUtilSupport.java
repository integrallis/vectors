package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pure-scalar reference implementation of {@link VectorUtilSupport}. Uses 4x-unrolled loops for ILP
 * (instruction-level parallelism) following the Lucene pattern. This serves as both the fallback
 * when the Vector API is unavailable and the correctness baseline for SIMD tests.
 */
final class ScalarVectorUtilSupport implements VectorUtilSupport {

  @Override
  public float dotProduct(float[] a, float[] b) {
    return dotProduct(a, 0, b, 0, a.length);
  }

  @Override
  public float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
    float res = 0f;
    int i = 0;

    // 4-way unrolled accumulation for ILP
    if (length > 32) {
      float acc1 = 0f;
      float acc2 = 0f;
      float acc3 = 0f;
      float acc4 = 0f;
      int upperBound = length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        acc1 = MathUtil.fma(a[aOffset + i], b[bOffset + i], acc1);
        acc2 = MathUtil.fma(a[aOffset + i + 1], b[bOffset + i + 1], acc2);
        acc3 = MathUtil.fma(a[aOffset + i + 2], b[bOffset + i + 2], acc3);
        acc4 = MathUtil.fma(a[aOffset + i + 3], b[bOffset + i + 3], acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    // Scalar tail
    for (; i < length; i++) {
      res = MathUtil.fma(a[aOffset + i], b[bOffset + i], res);
    }
    return res;
  }

  @Override
  public float squareDistance(float[] a, float[] b) {
    return squareDistance(a, 0, b, 0, a.length);
  }

  @Override
  public float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length) {
    float res = 0f;
    int i = 0;

    // 8-element unrolled blocks
    for (; i + 7 < length; i += 8) {
      res += squareDistanceUnrolled8(a, aOffset + i, b, bOffset + i);
    }

    // Scalar tail
    for (; i < length; i++) {
      float diff = a[aOffset + i] - b[bOffset + i];
      res += diff * diff;
    }
    return res;
  }

  private static float squareDistanceUnrolled8(float[] a, int ai, float[] b, int bi) {
    float d0 = a[ai] - b[bi];
    float d1 = a[ai + 1] - b[bi + 1];
    float d2 = a[ai + 2] - b[bi + 2];
    float d3 = a[ai + 3] - b[bi + 3];
    float d4 = a[ai + 4] - b[bi + 4];
    float d5 = a[ai + 5] - b[bi + 5];
    float d6 = a[ai + 6] - b[bi + 6];
    float d7 = a[ai + 7] - b[bi + 7];
    return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
  }

  @Override
  public float cosine(float[] a, float[] b) {
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;

    for (int i = 0; i < a.length; i++) {
      sum = MathUtil.fma(a[i], b[i], sum);
      norm1 = MathUtil.fma(a[i], a[i], norm1);
      norm2 = MathUtil.fma(b[i], b[i], norm2);
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int res = 0;
    for (int i = 0; i < a.length; i++) {
      res += a[i] * b[i];
    }
    return res;
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    int res = 0;
    for (int i = 0; i < a.length; i++) {
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

    for (int i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
      norm1 += a[i] * a[i];
      norm2 += b[i] * b[i];
    }

    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public float dotProduct(MemorySegment a, MemorySegment b, int dimensions) {
    float res = 0f;
    for (int i = 0; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      res = MathUtil.fma(ai, bi, res);
    }
    return res;
  }

  @Override
  public float squareDistance(MemorySegment a, MemorySegment b, int dimensions) {
    float res = 0f;
    for (int i = 0; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float diff = ai - bi;
      res += diff * diff;
    }
    return res;
  }

  @Override
  public int hammingDistance(long[] a, long[] b) {
    int dist = 0;
    for (int i = 0; i < a.length; i++) {
      dist += Long.bitCount(a[i] ^ b[i]);
    }
    return dist;
  }

  @Override
  public void addInPlace(float[] v1, float[] v2) {
    for (int i = 0; i < v1.length; i++) {
      v1[i] += v2[i];
    }
  }

  @Override
  public void subInPlace(float[] v1, float[] v2) {
    for (int i = 0; i < v1.length; i++) {
      v1[i] -= v2[i];
    }
  }

  @Override
  public void scale(float[] vector, float multiplier) {
    for (int i = 0; i < vector.length; i++) {
      vector[i] *= multiplier;
    }
  }

  @Override
  public float sum(float[] vector) {
    float res = 0f;
    for (float v : vector) {
      res += v;
    }
    return res;
  }

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
    scale(v, invNorm);
    return v;
  }
}
