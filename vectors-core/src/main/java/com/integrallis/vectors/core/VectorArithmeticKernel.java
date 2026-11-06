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
 * In-place vector arithmetic operations: addition, subtraction, scaling, element-wise fused
 * operations, and normalization.
 */
public interface VectorArithmeticKernel {

  /** Element-wise addition: {@code v1[i] += v2[i]}. */
  void addInPlace(float[] v1, float[] v2);

  /** Element-wise subtraction: {@code v1[i] -= v2[i]}. */
  void subInPlace(float[] v1, float[] v2);

  /** In-place scalar multiplication: {@code vector[i] *= multiplier}. */
  void scale(float[] vector, float multiplier);

  /** Sum of all elements. */
  float sum(float[] vector);

  /** SwiGLU activation: {@code silu(x) * y} where silu(x) = x / (1 + e^-x). */
  default void swiGlu(float[] silu, float[] gate) {
    for (int i = 0; i < silu.length; i++) {
      float x = silu[i];
      float sigmoid = x / (1.0f + (float) Math.exp(-x));
      silu[i] = sigmoid * gate[i];
    }
  }

  /** Sigmoid plus scaled softplus: {@code sigmoid(x) * softplus(y) * scale}. */
  default void sigmoidAndScaledSoftplus(float[] out, float[] x, float[] y, float scale) {
    for (int i = 0; i < out.length; i++) {
      float xv = x[i];
      float sigmoid = 1.0f / (1.0f + (float) Math.exp(-xv));
      float yv = y[i];
      float softplus =
          yv >= 0 ? yv + (float) Math.log1p(Math.exp(-yv)) : (float) Math.log1p(Math.exp(yv));
      out[i] = sigmoid * softplus * scale;
    }
  }

  /**
   * Causal depthwise 1-D convolution followed by SiLU: {@code conv1d(x, kernel) * sigmoid(conv1d(x,
   * kernel))}.
   */
  default void causalDepthwiseConv1dSilu(
      float[] out, float[] input, float[] kernel, int width, int groups) {
    int elementsPerGroup = width / groups;
    for (int g = 0; g < groups; g++) {
      int groupOffset = g * elementsPerGroup;
      for (int t = 0; t < elementsPerGroup; t++) {
        float accum = 0.0f;
        int kLen = kernel.length / groups;
        int kOffset = g * kLen;
        for (int k = 0; k <= t && k < kLen; k++) {
          accum += input[groupOffset + t - k] * kernel[kOffset + k];
        }
        float sigmoid = accum / (1.0f + (float) Math.exp(-accum));
        out[groupOffset + t] = accum * sigmoid;
      }
    }
  }

  /** Element-wise: {@code target[i] += a[i] * scaleA + b[i] * scaleB}. */
  default void addWeightedRowsInPlace(
      float[] target, float[] a, float scaleA, float[] b, float scaleB) {
    for (int i = 0; i < target.length; i++) {
      target[i] += a[i] * scaleA + b[i] * scaleB;
    }
  }

  /**
   * Normalizes a vector to unit L2 length. When {@code throwOnZero} is {@code true}, a zero vector
   * throws {@link IllegalArgumentException}; when {@code false}, it is returned unchanged.
   */
  float[] l2normalize(float[] v, boolean throwOnZero);
}
