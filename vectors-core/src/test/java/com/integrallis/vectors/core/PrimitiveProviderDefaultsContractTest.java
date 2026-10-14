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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** An extension implementing only required primitives inherits working matrix/vector contracts. */
class PrimitiveProviderDefaultsContractTest {
  @ParameterizedTest
  @ValueSource(ints = {1, 3, 4, 7, 16, 37, 257})
  void minimalProviderDefaultsPreserveDenseAlgebraAndOutputBounds(int n) {
    VectorUtilSupport provider = new PrimitiveProvider();
    float[] query = new float[n], row = new float[n], flat = new float[n * 2];
    double dot = 0, norm = 0, distance = 0, rowNorm = 0;
    for (int i = 0; i < n; i++) {
      query[i] = (i % 7 + 1) * .25f;
      row[i] = (i % 11 - 4) * .5f;
      flat[i] = row[i];
      flat[n + i] = query[i];
      dot += query[i] * row[i];
      norm += query[i] * query[i];
      rowNorm += row[i] * row[i];
      double d = query[i] - row[i];
      distance += d * d;
    }
    float[][] rows = {row, query};
    float[] out = {-991, -991, -991};
    provider.matVecDot(query, rows, out, 2);
    assertThat(out).containsExactly((float) dot, (float) norm, -991);
    Arrays.fill(out, -991);
    provider.matVecDot(query, flat, 2, n, out);
    assertThat(out).containsExactly((float) dot, (float) norm, -991);
    Arrays.fill(out, -991);
    provider.matVecSquaredL2(query, rows, out, 2);
    assertThat(out).containsExactly((float) distance, 0, -991);
    Arrays.fill(out, -991);
    provider.batchCosine(query, rows, out, 2);
    assertThat(out[0]).isCloseTo((float) (dot / Math.sqrt(norm * rowNorm)), within(1e-6f));
    assertThat(out[1]).isCloseTo(1f, within(1e-6f));
    assertThat(out[2]).isEqualTo(-991);
    float[] destination = new float[n + 6], source = new float[n + 8];
    Arrays.fill(destination, -991);
    Arrays.fill(source, 777);
    System.arraycopy(query, 0, destination, 2, n);
    System.arraycopy(row, 0, source, 3, n);
    float[] before = source.clone();
    provider.addScaledInPlace(destination, 2, source, 3, n, .5f);
    for (int i = 0; i < n; i++) assertThat(destination[i + 2]).isEqualTo(query[i] + .5f * row[i]);
    assertThat(Arrays.copyOf(destination, 2)).containsOnly(-991);
    assertThat(Arrays.copyOfRange(destination, n + 2, destination.length)).containsOnly(-991);
    assertThat(source).containsExactly(before);
  }

  private static final class PrimitiveProvider implements VectorUtilSupport {
    @Override
    public void swiGlu(
        float[] out,
        int outOffset,
        float[] gate,
        int gateOffset,
        float[] up,
        int upOffset,
        int length) {
      scalar.swiGlu(out, outOffset, gate, gateOffset, up, upOffset, length);
    }

    @Override
    public void sigmoidAndScaledSoftplus(
        float[] sigmoidValues,
        int sigmoidOffset,
        float[] softplusValues,
        int softplusOffset,
        float[] bias,
        int biasOffset,
        float[] scale,
        int scaleOffset,
        float[] scaledSoftplus,
        int scaledSoftplusOffset,
        int length) {
      scalar.sigmoidAndScaledSoftplus(
          sigmoidValues,
          sigmoidOffset,
          softplusValues,
          softplusOffset,
          bias,
          biasOffset,
          scale,
          scaleOffset,
          scaledSoftplus,
          scaledSoftplusOffset,
          length);
    }

    @Override
    public void causalDepthwiseConv1dSilu(
        float[] values,
        int valuesOffset,
        float[] history,
        int historyOffset,
        float[] weights,
        int weightsOffset,
        int channels,
        int kernelSize) {
      scalar.causalDepthwiseConv1dSilu(
          values,
          valuesOffset,
          history,
          historyOffset,
          weights,
          weightsOffset,
          channels,
          kernelSize);
    }

    @Override
    public void addWeightedRowsInPlace(
        float[] out,
        int outOffset,
        float[] matrix,
        int matrixOffset,
        int rowStride,
        float[] weights,
        int weightsOffset,
        int rows,
        int columns) {
      scalar.addWeightedRowsInPlace(
          out, outOffset, matrix, matrixOffset, rowStride, weights, weightsOffset, rows, columns);
    }

    @Override
    public void matVecDotExact(
        float[] query,
        int queryOffset,
        float[] matrix,
        int matrixOffset,
        int rowStride,
        int rows,
        int columns,
        float[] out,
        int outOffset) {
      scalar.matVecDotExact(
          query, queryOffset, matrix, matrixOffset, rowStride, rows, columns, out, outOffset);
    }

    private final VectorUtilSupport scalar = VectorizationProvider.newScalarProvider();

    @Override
    public float dotProduct(float[] a, float[] b) {
      return scalar.dotProduct(a, b);
    }

    @Override
    public float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
      return scalar.dotProduct(a, aOffset, b, bOffset, length);
    }

    @Override
    public float squareDistance(float[] a, float[] b) {
      return scalar.squareDistance(a, b);
    }

    @Override
    public float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length) {
      return scalar.squareDistance(a, aOffset, b, bOffset, length);
    }

    @Override
    public float cosine(float[] a, float[] b) {
      return scalar.cosine(a, b);
    }

    @Override
    public int dotProduct(byte[] a, byte[] b) {
      return scalar.dotProduct(a, b);
    }

    @Override
    public int squareDistance(byte[] a, byte[] b) {
      return scalar.squareDistance(a, b);
    }

    @Override
    public float cosine(byte[] a, byte[] b) {
      return scalar.cosine(a, b);
    }

    @Override
    public float dotProduct(MemorySegment a, MemorySegment b, int dimensions) {
      return scalar.dotProduct(a, b, dimensions);
    }

    @Override
    public float squareDistance(MemorySegment a, MemorySegment b, int dimensions) {
      return scalar.squareDistance(a, b, dimensions);
    }

    @Override
    public float cosine(MemorySegment a, MemorySegment b, int dimensions) {
      return scalar.cosine(a, b, dimensions);
    }

    @Override
    public int hammingDistance(long[] a, long[] b) {
      return scalar.hammingDistance(a, b);
    }

    @Override
    public void addInPlace(float[] v1, float[] v2) {
      scalar.addInPlace(v1, v2);
    }

    @Override
    public void subInPlace(float[] v1, float[] v2) {
      scalar.subInPlace(v1, v2);
    }

    @Override
    public void scale(float[] vector, float multiplier) {
      scalar.scale(vector, multiplier);
    }

    @Override
    public float sum(float[] vector) {
      return scalar.sum(vector);
    }

    @Override
    public float[] l2normalize(float[] v, boolean throwOnZero) {
      return scalar.l2normalize(v, throwOnZero);
    }
  }
}
