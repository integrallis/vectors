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
    int i = 0;

    // 4-way unrolled for ILP (3 FMAs per element, still benefits from multiple accumulators)
    if (a.length > 16) {
      float sum1 = 0f, sum2 = 0f, sum3 = 0f, sum4 = 0f;
      float n1a = 0f, n1b = 0f, n1c = 0f, n1d = 0f;
      float n2a = 0f, n2b = 0f, n2c = 0f, n2d = 0f;
      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        sum1 = MathUtil.fma(a[i], b[i], sum1);
        n1a = MathUtil.fma(a[i], a[i], n1a);
        n2a = MathUtil.fma(b[i], b[i], n2a);

        sum2 = MathUtil.fma(a[i + 1], b[i + 1], sum2);
        n1b = MathUtil.fma(a[i + 1], a[i + 1], n1b);
        n2b = MathUtil.fma(b[i + 1], b[i + 1], n2b);

        sum3 = MathUtil.fma(a[i + 2], b[i + 2], sum3);
        n1c = MathUtil.fma(a[i + 2], a[i + 2], n1c);
        n2c = MathUtil.fma(b[i + 2], b[i + 2], n2c);

        sum4 = MathUtil.fma(a[i + 3], b[i + 3], sum4);
        n1d = MathUtil.fma(a[i + 3], a[i + 3], n1d);
        n2d = MathUtil.fma(b[i + 3], b[i + 3], n2d);
      }
      sum = sum1 + sum2 + sum3 + sum4;
      norm1 = n1a + n1b + n1c + n1d;
      norm2 = n2a + n2b + n2c + n2d;
    }

    // Scalar tail
    for (; i < a.length; i++) {
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
  public float cosine(MemorySegment a, MemorySegment b, int dimensions) {
    float sum = 0f;
    float norm1 = 0f;
    float norm2 = 0f;
    for (int i = 0; i < dimensions; i++) {
      float ai = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      float bi = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      sum = MathUtil.fma(ai, bi, sum);
      norm1 = MathUtil.fma(ai, ai, norm1);
      norm2 = MathUtil.fma(bi, bi, norm2);
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
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
  public void addScaledInPlace(
      float[] out, int outOffset, float[] vector, int vectorOffset, int length, float scale) {
    int i = 0;
    int upperBound = length & ~3;
    for (; i < upperBound; i += 4) {
      int outIndex = outOffset + i;
      int vectorIndex = vectorOffset + i;
      out[outIndex] = MathUtil.fma(vector[vectorIndex], scale, out[outIndex]);
      out[outIndex + 1] = MathUtil.fma(vector[vectorIndex + 1], scale, out[outIndex + 1]);
      out[outIndex + 2] = MathUtil.fma(vector[vectorIndex + 2], scale, out[outIndex + 2]);
      out[outIndex + 3] = MathUtil.fma(vector[vectorIndex + 3], scale, out[outIndex + 3]);
    }
    for (; i < length; i++) {
      int outIndex = outOffset + i;
      out[outIndex] = MathUtil.fma(vector[vectorOffset + i], scale, out[outIndex]);
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

  // --- Fused batch matrix-vector kernels (GEMV) ---

  /**
   * Scalar 4-row-unrolled GEMV for dot product.
   *
   * <p>Processes 4 rows at a time in the outer loop. In the inner dimension loop each iteration
   * accumulates one element into each of the 4 per-row accumulators simultaneously, giving 4
   * independent FMA chains for ILP even in the scalar JIT tier.
   */
  @Override
  public void matVecDot(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3; // round down to multiple of 4

    // Main: groups of 4 rows
    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;
      int i = 0;
      int bound = dim & ~3;
      for (; i < bound; i += 4) {
        float q0 = query[i], q1 = query[i + 1], q2 = query[i + 2], q3 = query[i + 3];
        a0 =
            MathUtil.fma(
                q0,
                r0[i],
                MathUtil.fma(
                    q1, r0[i + 1], MathUtil.fma(q2, r0[i + 2], MathUtil.fma(q3, r0[i + 3], a0))));
        a1 =
            MathUtil.fma(
                q0,
                r1[i],
                MathUtil.fma(
                    q1, r1[i + 1], MathUtil.fma(q2, r1[i + 2], MathUtil.fma(q3, r1[i + 3], a1))));
        a2 =
            MathUtil.fma(
                q0,
                r2[i],
                MathUtil.fma(
                    q1, r2[i + 1], MathUtil.fma(q2, r2[i + 2], MathUtil.fma(q3, r2[i + 3], a2))));
        a3 =
            MathUtil.fma(
                q0,
                r3[i],
                MathUtil.fma(
                    q1, r3[i + 1], MathUtil.fma(q2, r3[i + 2], MathUtil.fma(q3, r3[i + 3], a3))));
      }
      for (; i < dim; i++) {
        float q = query[i];
        a0 = MathUtil.fma(q, r0[i], a0);
        a1 = MathUtil.fma(q, r1[i], a1);
        a2 = MathUtil.fma(q, r2[i], a2);
        a3 = MathUtil.fma(q, r3[i], a3);
      }
      out[r] = a0;
      out[r + 1] = a1;
      out[r + 2] = a2;
      out[r + 3] = a3;
    }

    // Tail: remaining 1-3 rows
    for (int r = rowGroup; r < numRows; r++) {
      out[r] = dotProduct(query, 0, matrix[r], 0, dim);
    }
  }

  /** Scalar 4-row-unrolled GEMV for squared L2 distance. */
  @Override
  public void matVecSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    int dim = query.length;
    int rowGroup = numRows & ~3;

    for (int r = 0; r < rowGroup; r += 4) {
      float[] r0 = matrix[r], r1 = matrix[r + 1], r2 = matrix[r + 2], r3 = matrix[r + 3];
      float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;
      int i = 0;
      int bound = dim & ~3;
      for (; i < bound; i += 4) {
        float q0 = query[i], q1 = query[i + 1], q2 = query[i + 2], q3 = query[i + 3];
        float d0, d1, d2, d3;
        d0 = q0 - r0[i];
        a0 = MathUtil.fma(d0, d0, a0);
        d0 = q0 - r1[i];
        a1 = MathUtil.fma(d0, d0, a1);
        d0 = q0 - r2[i];
        a2 = MathUtil.fma(d0, d0, a2);
        d0 = q0 - r3[i];
        a3 = MathUtil.fma(d0, d0, a3);
        d1 = q1 - r0[i + 1];
        a0 = MathUtil.fma(d1, d1, a0);
        d1 = q1 - r1[i + 1];
        a1 = MathUtil.fma(d1, d1, a1);
        d1 = q1 - r2[i + 1];
        a2 = MathUtil.fma(d1, d1, a2);
        d1 = q1 - r3[i + 1];
        a3 = MathUtil.fma(d1, d1, a3);
        d2 = q2 - r0[i + 2];
        a0 = MathUtil.fma(d2, d2, a0);
        d2 = q2 - r1[i + 2];
        a1 = MathUtil.fma(d2, d2, a1);
        d2 = q2 - r2[i + 2];
        a2 = MathUtil.fma(d2, d2, a2);
        d2 = q2 - r3[i + 2];
        a3 = MathUtil.fma(d2, d2, a3);
        d3 = q3 - r0[i + 3];
        a0 = MathUtil.fma(d3, d3, a0);
        d3 = q3 - r1[i + 3];
        a1 = MathUtil.fma(d3, d3, a1);
        d3 = q3 - r2[i + 3];
        a2 = MathUtil.fma(d3, d3, a2);
        d3 = q3 - r3[i + 3];
        a3 = MathUtil.fma(d3, d3, a3);
      }
      for (; i < dim; i++) {
        float q = query[i];
        float d0 = q - r0[i];
        a0 = MathUtil.fma(d0, d0, a0);
        float d1 = q - r1[i];
        a1 = MathUtil.fma(d1, d1, a1);
        float d2 = q - r2[i];
        a2 = MathUtil.fma(d2, d2, a2);
        float d3 = q - r3[i];
        a3 = MathUtil.fma(d3, d3, a3);
      }
      out[r] = a0;
      out[r + 1] = a1;
      out[r + 2] = a2;
      out[r + 3] = a3;
    }

    for (int r = rowGroup; r < numRows; r++) {
      out[r] = squareDistance(query, 0, matrix[r], 0, dim);
    }
  }
}
