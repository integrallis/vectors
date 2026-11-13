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
 * Similarity scoring functions that convert raw distance/similarity values to a normalized score
 * where higher values indicate greater similarity.
 *
 * <p>For <b>float</b> vectors, scores are normalized to a non-negative range. For <b>byte</b>
 * vectors, scores are also transformed, but the specific transform depends on the metric and
 * encoding (see individual enum members for formulas and ranges). The class-level claim that byte
 * scores are returned raw is <b>incorrect</b> — byte EUCLIDEAN, COSINE, and MAXIMUM_INNER_PRODUCT
 * use the same transform as their float counterparts, and byte DOT_PRODUCT uses a
 * dimension-dependent scaling.
 *
 * <p>All transforms are monotonic with respect to the underlying raw value, so ranking and recall
 * are preserved even though the absolute score values differ from the raw metric.
 *
 * <p>Modeled after Lucene's {@code VectorSimilarityFunction} and JVector's {@code
 * VectorSimilarityFunction}.
 */
public enum SimilarityFunction {

  /**
   * Euclidean distance: higher scores for closer vectors.
   *
   * <p>Float vector formula: {@code score = 1 / (1 + squareDistance)}. Range: {@code (0, 1]} where
   * 1 means identical.
   *
   * <p>Byte vector formula: same as float ({@code 1 / (1 + squareDistance)}), so byte scores are
   * also transformed. Range: {@code (0, 1]}.
   */
  EUCLIDEAN {
    @Override
    public float compare(float[] v1, float[] v2) {
      return 1f / (1f + VectorUtil.squareDistance(v1, v2));
    }

    @Override
    public float compare(byte[] v1, byte[] v2) {
      return 1f / (1f + VectorUtil.squareDistance(v1, v2));
    }

    @Override
    public float compare(MemorySegment v1, MemorySegment v2, int dimensions) {
      return 1f / (1f + VectorUtil.squareDistance(v1, v2, dimensions));
    }
  },

  /**
   * Dot product similarity. Assumes float vectors are normalized (unit length).
   *
   * <p>Float vector formula: {@code score = (1 + dotProduct) / 2}. Range: {@code [0, 1]} for
   * normalized vectors.
   *
   * <p>Byte vector formula: {@code score = 0.5 + dot / (2 * dim * 127 * 127)}. This scales the raw
   * byte dot product by the maximum possible magnitude ({@code 127 * 127 * dim}) so the result is
   * centered at 0.5 and remains in {@code [0, 1]}. Not identical to the float transform, but
   * preserves monotonic order.
   */
  DOT_PRODUCT {
    @Override
    public float compare(float[] v1, float[] v2) {
      return (1f + VectorUtil.dotProduct(v1, v2)) / 2f;
    }

    @Override
    public float compare(byte[] v1, byte[] v2) {
      float dot = VectorUtil.dotProduct(v1, v2);
      // Byte vectors: scale by max possible magnitude (127*127*dims)
      return 0.5f + dot / (float) (2 * v1.length * 127 * 127);
    }

    @Override
    public float compare(MemorySegment v1, MemorySegment v2, int dimensions) {
      return (1f + VectorUtil.dotProduct(v1, v2, dimensions)) / 2f;
    }
  },

  /**
   * Cosine similarity: direction-based similarity ignoring magnitude.
   *
   * <p>Float vector formula: {@code score = (1 + cosine) / 2}. Range: {@code [0, 1]} where 1 means
   * same direction.
   *
   * <p>Byte vector formula: identical to float ({@code (1 + cosine) / 2}). Range: {@code [0, 1]}.
   */
  COSINE {
    @Override
    public float compare(float[] v1, float[] v2) {
      return (1f + VectorUtil.cosine(v1, v2)) / 2f;
    }

    @Override
    public float compare(byte[] v1, byte[] v2) {
      return (1f + VectorUtil.cosine(v1, v2)) / 2f;
    }

    @Override
    public float compare(MemorySegment v1, MemorySegment v2, int dimensions) {
      return (1f + VectorUtil.cosine(v1, v2, dimensions)) / 2f;
    }
  },

  /**
   * Maximum inner product: for non-normalized vectors where larger dot products are better.
   *
   * <p>Both float and byte vectors use the same piecewise transform. Formula:
   *
   * <ul>
   *   <li>If {@code dot >= 0}: {@code score = dot + 1}
   *   <li>If {@code dot < 0}: {@code score = 1 / (1 - dot)}
   * </ul>
   *
   * The piecewise function ensures monotonic ordering (larger raw dot products always produce
   * larger scores) and maps to a non-negative range.
   */
  MAXIMUM_INNER_PRODUCT {
    @Override
    public float compare(float[] v1, float[] v2) {
      float dot = VectorUtil.dotProduct(v1, v2);
      return scaleMaxInnerProductScore(dot);
    }

    @Override
    public float compare(byte[] v1, byte[] v2) {
      float dot = VectorUtil.dotProduct(v1, v2);
      return scaleMaxInnerProductScore(dot);
    }

    @Override
    public float compare(MemorySegment v1, MemorySegment v2, int dimensions) {
      float dot = VectorUtil.dotProduct(v1, v2, dimensions);
      return scaleMaxInnerProductScore(dot);
    }
  };

  /**
   * Computes the similarity score between two float vectors. Higher values indicate greater
   * similarity.
   */
  public abstract float compare(float[] v1, float[] v2);

  /**
   * Computes the similarity score between two byte vectors. Higher values indicate greater
   * similarity.
   */
  public abstract float compare(byte[] v1, byte[] v2);

  /**
   * Computes the similarity score between two float vectors stored off-heap in {@link
   * MemorySegment}s. Higher values indicate greater similarity.
   *
   * <p>This is the zero-copy scoring entry point: the segments are read directly by the {@link
   * VectorUtil} SIMD kernels via {@code FloatVector.fromMemorySegment} — no intermediate {@code
   * float[]} copy. Numerically it mirrors {@link #compare(float[], float[])} exactly (same
   * transform, same kernels, same float32 accumulation), so it preserves ranking/recall.
   *
   * @param v1 first vector as an off-heap segment holding {@code dimensions} little-endian float32s
   * @param v2 second vector as an off-heap segment holding {@code dimensions} little-endian
   *     float32s
   * @param dimensions the number of float elements in each segment
   * @return the similarity score (higher means more similar)
   */
  public abstract float compare(MemorySegment v1, MemorySegment v2, int dimensions);

  /**
   * Scales a raw inner product score to a non-negative value. Piecewise function ensures monotonic
   * ordering: larger dot products always produce larger scores.
   */
  public static float scaleMaxInnerProductScore(float dotProduct) {
    if (dotProduct >= 0) {
      return dotProduct + 1f;
    } else {
      return 1f / (1f - dotProduct);
    }
  }
}
