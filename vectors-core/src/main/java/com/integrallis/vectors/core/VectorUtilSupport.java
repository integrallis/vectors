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

/**
 * Interface for all SIMD-accelerable vector operations. Implementations are selected at runtime by
 * {@link VectorizationProvider}: Panama Vector API (preferred) or scalar fallback.
 *
 * <p>Modeled after Apache Lucene's {@code VectorUtilSupport} and JVector's {@code
 * VectorUtilSupport}, combining the best patterns from both.
 */
public interface VectorUtilSupport {

  // --- Float distance kernels ---

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

  // --- Byte distance kernels ---

  /** Computes the dot product of two signed byte vectors. */
  int dotProduct(byte[] a, byte[] b);

  /** Computes the squared Euclidean distance between two signed byte vectors. */
  int squareDistance(byte[] a, byte[] b);

  /** Computes the cosine similarity between two signed byte vectors. */
  float cosine(byte[] a, byte[] b);

  // --- MemorySegment distance kernels (off-heap) ---

  /** Computes the dot product from off-heap float vectors stored in MemorySegments. */
  float dotProduct(MemorySegment a, MemorySegment b, int dimensions);

  /** Computes the squared L2 distance from off-heap float vectors stored in MemorySegments. */
  float squareDistance(MemorySegment a, MemorySegment b, int dimensions);

  /**
   * Computes the cosine similarity from off-heap float vectors stored in MemorySegments. Returns
   * value in [-1, 1].
   */
  float cosine(MemorySegment a, MemorySegment b, int dimensions);

  // --- Binary distance ---

  /** Computes the Hamming distance between two binary vectors stored as packed long arrays. */
  int hammingDistance(long[] a, long[] b);

  // --- Vector arithmetic ---

  /** Adds v2 to v1 element-wise, storing the result in v1. */
  void addInPlace(float[] v1, float[] v2);

  /** Subtracts v2 from v1 element-wise, storing the result in v1. */
  void subInPlace(float[] v1, float[] v2);

  /** Scales each element of the vector by the given multiplier. */
  void scale(float[] vector, float multiplier);

  /** Returns the sum of all elements in the vector. */
  float sum(float[] vector);

  // --- Normalization ---

  /**
   * L2-normalizes the given vector in place and returns it.
   *
   * @param v the vector to normalize
   * @param throwOnZero if true, throws IllegalArgumentException for zero-length vectors
   * @return the normalized vector (same array reference)
   */
  float[] l2normalize(float[] v, boolean throwOnZero);

  // --- Fused batch matrix-vector kernels (GEMV) ---

  /**
   * Fused matrix-vector dot product: fills {@code out[i] = dot(query, matrix[i])} for {@code i in
   * [0, numRows)}.
   *
   * <p>The default implementation is a simple loop. SIMD subclasses override this to load each
   * query SIMD chunk <em>once</em> and apply it to 4 matrix rows simultaneously, cutting query
   * memory traffic by 4× compared to calling {@link #dotProduct} per row.
   *
   * @param query the query vector (length = {@code matrix[0].length})
   * @param matrix the matrix rows (each row must have the same length as {@code query})
   * @param out the output array (must have length &ge; {@code numRows})
   * @param numRows the number of rows to process (must be &le; {@code matrix.length})
   */
  default void matVecDot(float[] query, float[][] matrix, float[] out, int numRows) {
    for (int i = 0; i < numRows; i++) {
      out[i] = dotProduct(query, 0, matrix[i], 0, query.length);
    }
  }

  /**
   * Fused matrix-vector squared L2 distance: fills {@code out[i] = squaredL2(query, matrix[i])} for
   * {@code i in [0, numRows)}.
   *
   * <p>The default implementation is a simple loop. SIMD subclasses override this with 4-row
   * unrolled accumulation.
   *
   * @param query the query vector
   * @param matrix the matrix rows
   * @param out the output array (must have length &ge; {@code numRows})
   * @param numRows the number of rows to process
   */
  default void matVecSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    for (int i = 0; i < numRows; i++) {
      out[i] = squareDistance(query, 0, matrix[i], 0, query.length);
    }
  }

  // --- PQ ADC (Asymmetric Distance Computation) kernels ---

  /**
   * Sums one entry per subspace from a precomputed query→centroid table, indexed by PQ codes.
   *
   * <p>Computes {@code sum_{m=0..M-1} table[m][codes[codesOffset + m] & 0xFF]} — the core inner
   * loop of ADC scoring against PQ-compressed vectors. Equivalent to the lookup {@code T[m][c_m]}
   * pattern used by JVector's {@code VectorUtil.assembleAndSum}.
   *
   * @param table precomputed table of shape {@code [numSubspaces][numClusters]}; typically produced
   *     by a product-quantizer ADC table builder
   * @param codes byte buffer holding at least {@code numSubspaces} PQ codes starting at {@code
   *     codesOffset} (unsigned)
   * @param codesOffset starting offset in {@code codes}
   * @param numSubspaces M — the number of PQ subspaces
   * @return the accumulated partial distance/similarity (unscored raw sum)
   */
  default float assembleAndSum(float[][] table, byte[] codes, int codesOffset, int numSubspaces) {
    float sum = 0f;
    for (int m = 0; m < numSubspaces; m++) {
      sum += table[m][codes[codesOffset + m] & 0xFF];
    }
    return sum;
  }

  /**
   * Batched variant of {@link #assembleAndSum} over {@code count} neighbors stored in a packed byte
   * buffer. Each neighbor occupies {@code numSubspaces} consecutive bytes: neighbor {@code i}
   * starts at {@code codesOffset + i * numSubspaces}.
   *
   * <p>The 4-row unrolled default keeps each {@code table[m]} pointer hot across four lookups,
   * amortising the subspace-table load and eliminating three redundant table row references per
   * iteration vs the scalar scorer.
   */
  default void batchAssembleAndSum(
      float[][] table,
      byte[] packedCodes,
      int codesOffset,
      float[] out,
      int count,
      int numSubspaces) {
    int i = 0;
    int m;
    int rowGroup = count & ~3;
    for (; i < rowGroup; i += 4) {
      float s0 = 0f;
      float s1 = 0f;
      float s2 = 0f;
      float s3 = 0f;
      int o0 = codesOffset + i * numSubspaces;
      int o1 = o0 + numSubspaces;
      int o2 = o1 + numSubspaces;
      int o3 = o2 + numSubspaces;
      for (m = 0; m < numSubspaces; m++) {
        float[] tm = table[m];
        s0 += tm[packedCodes[o0 + m] & 0xFF];
        s1 += tm[packedCodes[o1 + m] & 0xFF];
        s2 += tm[packedCodes[o2 + m] & 0xFF];
        s3 += tm[packedCodes[o3 + m] & 0xFF];
      }
      out[i] = s0;
      out[i + 1] = s1;
      out[i + 2] = s2;
      out[i + 3] = s3;
    }
    for (; i < count; i++) {
      float s = 0f;
      int o = codesOffset + i * numSubspaces;
      for (m = 0; m < numSubspaces; m++) {
        s += table[m][packedCodes[o + m] & 0xFF];
      }
      out[i] = s;
    }
  }
}
