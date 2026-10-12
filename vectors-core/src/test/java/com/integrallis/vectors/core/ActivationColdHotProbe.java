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

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;

/**
 * Checks public activation outputs and recurrent history across compilation and concurrent callers.
 */
final class ActivationColdHotProbe {
  public static void main(String[] args) throws Exception {
    check(4021);
    try (var executor = Executors.newFixedThreadPool(4)) {
      var work = new java.util.ArrayList<java.util.concurrent.Callable<Void>>();
      for (int thread = 0; thread < 4; thread++) {
        long seed = 4022 + thread;
        work.add(
            () -> {
              check(seed);
              return null;
            });
      }
      for (var result : executor.invokeAll(work)) result.get();
    }
    System.out.println(
        "ACTIVATIONS_STABLE cold/hot, offsets, tails, aliasing, history, concurrent callers");
  }

  private static void check(long seed) {
    int n = 259, offset = 3;
    SplittableRandom random = new SplittableRandom(seed);
    float[] gate = random(random, n + 2 * offset), up = random(random, n + 2 * offset);
    float[] bias = random(random, n + 2 * offset), scale = random(random, n + 2 * offset);
    float[] initialHistory = random(random, 2 * n + 2 * offset);
    float[] weights = random(random, 3 * n + 2 * offset);
    float[] sigmoid = new float[gate.length], softplus = new float[gate.length];
    float[] swiglu = new float[gate.length], convolution = new float[gate.length];
    float[] history = new float[initialHistory.length];
    float[][] first = null;
    for (int iteration = 0; iteration < 5000; iteration++) {
      System.arraycopy(gate, 0, sigmoid, 0, gate.length);
      System.arraycopy(up, 0, swiglu, 0, up.length);
      System.arraycopy(gate, 0, convolution, 0, gate.length);
      System.arraycopy(initialHistory, 0, history, 0, history.length);
      Arrays.fill(softplus, 123.0f);
      VectorUtil.swiGlu(swiglu, offset, gate, offset, swiglu, offset, n);
      VectorUtil.sigmoidAndScaledSoftplus(
          sigmoid, offset, up, offset, bias, offset, scale, offset, softplus, offset, n);
      VectorUtil.causalDepthwiseConv1dSilu(
          convolution, offset, history, offset, weights, offset, n, 3);
      float[][] current = {swiglu, sigmoid, softplus, convolution, history};
      if (first == null) {
        first = Arrays.stream(current).map(float[]::clone).toArray(float[][]::new);
      } else {
        for (int operation = 0; operation < current.length; operation++) {
          if (!Arrays.equals(first[operation], current[operation])) {
            int cell = Arrays.mismatch(first[operation], current[operation]);
            throw new AssertionError(
                "activation drift seed="
                    + seed
                    + " iteration="
                    + iteration
                    + " operation="
                    + operation
                    + " cell="
                    + cell
                    + " cold="
                    + first[operation][cell]
                    + " hot="
                    + current[operation][cell]);
          }
        }
      }
      if (swiglu[0] != up[0]
          || sigmoid[0] != gate[0]
          || softplus[0] != 123.0f
          || convolution[0] != gate[0]
          || history[0] != initialHistory[0]) {
        throw new AssertionError("prefix guard modified");
      }
    }
  }

  private static float[] random(SplittableRandom random, int n) {
    float[] values = new float[n];
    for (int i = 0; i < n; i++) values[i] = (float) random.nextDouble(-8, 8);
    return values;
  }
}
