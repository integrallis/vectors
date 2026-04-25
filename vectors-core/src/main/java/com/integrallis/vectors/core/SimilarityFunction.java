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
 * Similarity scoring functions that convert raw distance/similarity values to a normalized score
 * where higher values indicate greater similarity.
 *
 * <p>For float vectors, scores are normalized to a non-negative range. For byte vectors, the raw
 * score is returned (callers should normalize as needed).
 *
 * <p>Modeled after Lucene's {@code VectorSimilarityFunction} and JVector's {@code
 * VectorSimilarityFunction}.
 */
public enum SimilarityFunction {

  /**
   * Euclidean distance: higher scores for closer vectors. Score = 1 / (1 + squareDistance). Result
   * range: (0, 1] where 1 means identical.
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
  },

  /**
   * Dot product similarity: assumes vectors are normalized (unit length). Score = (1 + dotProduct)
   * / 2. Result range: [0, 1] for normalized vectors.
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
  },

  /**
   * Cosine similarity: direction-based similarity ignoring magnitude. Score = (1 + cosine) / 2.
   * Result range: [0, 1] where 1 means same direction.
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
  },

  /**
   * Maximum inner product: for non-normalized vectors where larger dot products are better. Score
   * uses a piecewise function to map to non-negative range: if dot >= 0: score = dot + 1; if dot
   * &lt; 0: score = 1 / (1 - dot).
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
