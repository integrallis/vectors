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

/** Actual 48-head gating and persistent four-tap history, independent of model weights. */
final class BonsaiActivationSequenceProbe {
  private static final int HEADS = 48, CHANNELS = 259, STEPS = 12, OFFSET = 3;

  public static void main(String[] args) {
    int bits = PanamaVectorUtilSupport.FLOAT_SPECIES.vectorBitSize();
    boolean panama = VectorizationProvider.isPanamaEnabled();
    System.out.printf(
        "CONFIG requested=%s effective=%d preferred=%d vectorFma=%s scalarFma=%s java=%s arch=%s%n",
        System.getProperty("vectors.maxBits", "default"),
        bits,
        PanamaConstants.PREFERRED_BITS,
        PanamaConstants.HAS_FAST_VECTOR_FMA,
        PanamaConstants.HAS_FAST_SCALAR_FMA,
        System.getProperty("java.runtime.version"),
        System.getProperty("os.arch"));
    int lanes = bits / Float.SIZE;
    SplittableRandom random = new SplittableRandom(938417);
    float[][] inputs = new float[STEPS][], betas = new float[STEPS][], alphas = new float[STEPS][];
    for (int step = 0; step < STEPS; step++) {
      inputs[step] = random(random, CHANNELS + 2 * OFFSET, 2);
      betas[step] = random(random, HEADS + 2 * OFFSET, 8);
      alphas[step] = random(random, HEADS + 2 * OFFSET, 8);
    }
    float[] bias = random(random, HEADS + 2 * OFFSET, 1);
    float[] scale = random(random, HEADS + 2 * OFFSET, 2);
    // Preserve the existing vector float-intermediate contract, including its negative tail.
    float[] edges = {-104, -21, -20, -17, -16, -0.0f, 0, 20, Math.nextUp(20f), 89};
    for (int i = 0; i < edges.length; i++) {
      alphas[0][OFFSET + i] = edges[i];
      bias[OFFSET + i] = 0;
    }
    float[] initialHistory = random(random, 3 * CHANNELS + 2 * OFFSET, .25f);
    float[] weights = random(random, 4 * CHANNELS + 2 * OFFSET, .25f);
    float[] values = new float[inputs[0].length], beta = new float[betas[0].length];
    float[] decay = new float[beta.length], history = initialHistory.clone();
    float[][][] cold = new float[STEPS][][];
    for (int iteration = 0; iteration < 2000; iteration++) {
      System.arraycopy(initialHistory, 0, history, 0, history.length);
      float[] referenceHistory = iteration == 0 ? initialHistory.clone() : null;
      for (int step = 0; step < STEPS; step++) {
        System.arraycopy(inputs[step], 0, values, 0, values.length);
        System.arraycopy(betas[step], 0, beta, 0, beta.length);
        Arrays.fill(decay, 123f);
        VectorUtil.sigmoidAndScaledSoftplus(
            beta, OFFSET, alphas[step], OFFSET, bias, OFFSET, scale, OFFSET, decay, OFFSET, HEADS);
        VectorUtil.causalDepthwiseConv1dSilu(
            values, OFFSET, history, OFFSET, weights, OFFSET, CHANNELS, 4);
        if (iteration == 0) {
          float[] expectedBeta = betas[step].clone(), expectedDecay = new float[decay.length];
          Arrays.fill(expectedDecay, 123f);
          for (int h = 0; h < HEADS; h++) {
            int i = OFFSET + h;
            float x = alphas[step][i] + bias[i];
            expectedBeta[i] = 1f / (1f + (float) StrictMath.exp(-expectedBeta[i]));
            float softplus =
                panama && h < HEADS - HEADS % lanes
                    ? (x > 20f ? x : (float) StrictMath.log(1f + (float) StrictMath.exp(x)))
                    : x > 20f
                        ? x
                        : x < -20f
                            ? (float) StrictMath.exp(x)
                            : (float) StrictMath.log1p(StrictMath.exp(x));
            expectedDecay[i] = scale[i] * softplus;
          }
          float[] expectedValues = inputs[step].clone();
          for (int channel = 0; channel < CHANNELS; channel++) {
            float input = inputs[step][OFFSET + channel];
            float sum = input * weights[OFFSET + 3 * CHANNELS + channel];
            for (int tap = 0; tap < 3; tap++) {
              int index = OFFSET + tap * CHANNELS + channel;
              float previous = referenceHistory[index];
              sum =
                  panama
                          && channel < CHANNELS - CHANNELS % lanes
                          && PanamaConstants.HAS_FAST_VECTOR_FMA
                      ? Math.fma(previous, weights[index], sum)
                      : previous * weights[index] + sum;
              if (tap > 0) referenceHistory[index - CHANNELS] = previous;
            }
            referenceHistory[OFFSET + 2 * CHANNELS + channel] = input;
            expectedValues[OFFSET + channel] = sum / (1f + (float) StrictMath.exp(-sum));
          }
          equal("sigmoid contract", expectedBeta, beta, iteration, step);
          equal("softplus float-intermediate contract", expectedDecay, decay, iteration, step);
          equal("convolution contract", expectedValues, values, iteration, step);
          equal("persistent history contract", referenceHistory, history, iteration, step);
          cold[step] = new float[][] {beta.clone(), decay.clone(), values.clone(), history.clone()};
        } else {
          equal("sigmoid cold/hot", cold[step][0], beta, iteration, step);
          equal("softplus cold/hot", cold[step][1], decay, iteration, step);
          equal("convolution cold/hot", cold[step][2], values, iteration, step);
          equal("persistent history cold/hot", cold[step][3], history, iteration, step);
        }
      }
    }
    System.out.println("BONSAI_ACTIVATIONS_STABLE heads=48 kernel=4 steps=12 iterations=2000");
  }

  private static float[] random(SplittableRandom random, int size, float magnitude) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++) values[i] = (float) random.nextDouble(-magnitude, magnitude);
    return values;
  }

  private static void equal(
      String operation, float[] expected, float[] actual, int iteration, int step) {
    for (int i = 0; i < expected.length; i++) {
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i])) {
        throw new AssertionError(
            operation
                + " iteration="
                + iteration
                + " step="
                + step
                + " cell="
                + i
                + " expected="
                + Float.toHexString(expected[i])
                + " actual="
                + Float.toHexString(actual[i]));
      }
    }
  }
}
