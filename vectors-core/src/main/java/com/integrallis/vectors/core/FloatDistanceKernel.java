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

/**
 * Float-array distance and similarity kernels. The minimal interface for dense float-vector
 * operations: dot product, squared Euclidean distance, and cosine similarity.
 *
 * <p>Implementations are expected to be SIMD-accelerated where hardware permits. All methods
 * require matching dimensions and throw {@link IllegalArgumentException} when vectors differ in
 * length.
 */
public interface FloatDistanceKernel {

  /** Computes the dot product of two float vectors. */
  float dotProduct(float[] a, float[] b);

  /** Computes the dot product of sub-vectors at the given offsets. */
  float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Computes the squared Euclidean (L2) distance between two float vectors. */
  float squareDistance(float[] a, float[] b);

  /** Computes the squared Euclidean distance of sub-vectors at the given offsets. */
  float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Computes the cosine similarity between two float vectors. Returns value in [-1, 1]. */
  float cosine(float[] a, float[] b);

  /** Whether this provider can reuse both norms without changing cosine arithmetic. */
  default boolean supportsCosineNormReuse(int dimensions) {
    return false;
  }

  /** Prepares a query norm for repeated cosine comparisons. */
  default float cosineQueryNorm(float[] query) {
    return dotProduct(query, query);
  }

  /** Cosine with a norm prepared by this provider; scalar providers retain their arithmetic. */
  default float cosineWithQueryNorm(float[] a, float[] b, float queryNorm) {
    return cosine(a, b);
  }

  /** Cosine with both norms prepared by this provider. */
  default float cosineWithNorms(float[] a, float[] b, float norm1, float norm2) {
    return cosine(a, b);
  }
}
