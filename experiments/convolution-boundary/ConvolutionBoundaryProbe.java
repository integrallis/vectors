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

import java.lang.management.ManagementFactory;
import java.util.SplittableRandom;

/** Standalone kernel diagnostic: no model inference, no mutable backend configuration. */
public final class ConvolutionBoundaryProbe {
  static volatile double sink;
  public static void main(String[] args) {
    final int channels = 10240, kernel = 4, warm = 1000, measured = 500;
    var allocations = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    allocations.setThreadAllocatedMemoryEnabled(true);
    long thread = Thread.currentThread().threadId();
    int lanes = PanamaVectorUtilSupport.FLOAT_SPECIES.length();
    boolean fma = PanamaConstants.HAS_FAST_VECTOR_FMA;
    float[] values = new float[channels], initial = new float[channels];
    float[] history = new float[channels * 3], weights = new float[channels * 4];
    SplittableRandom random = new SplittableRandom(810753);
    for (int i = 0; i < channels; i++) initial[i] = (float) random.nextDouble(-4, 4);
    for (int i = 0; i < history.length; i++) history[i] = (float) random.nextDouble(-1, 1);
    for (int i = 0; i < weights.length; i++) weights[i] = (float) random.nextDouble(-.5, .5);
    System.arraycopy(initial, 0, values, 0, channels);
    float[] oracleHistory = history.clone(), oracleValues = values.clone();
    long coldBytes = allocations.getThreadAllocatedBytes(thread), coldNanos = System.nanoTime();
    VectorUtil.causalDepthwiseConv1dSilu(values, 0, history, 0, weights, 0, channels, kernel);
    coldNanos = System.nanoTime() - coldNanos;
    coldBytes = allocations.getThreadAllocatedBytes(thread) - coldBytes;
    reference(oracleValues, oracleHistory, weights, lanes, fma);
    int coldMismatches = differences(values, oracleValues) + differences(history, oracleHistory);
    long hotNanos = 0, hotBytes = 0, maxBytes = 0;
    for (int iteration = 0; iteration < warm + measured; iteration++) {
      for (int i = 0; i < channels; i++) values[i] = initial[i] + (iteration % 17 - 8) * .00390625f;
      long beforeBytes = allocations.getThreadAllocatedBytes(thread), beforeNanos = System.nanoTime();
      VectorUtil.causalDepthwiseConv1dSilu(values, 0, history, 0, weights, 0, channels, kernel);
      long nanos = System.nanoTime() - beforeNanos;
      long bytes = allocations.getThreadAllocatedBytes(thread) - beforeBytes;
      if (iteration >= warm) { hotNanos += nanos; hotBytes += bytes; maxBytes = Math.max(maxBytes, bytes); }
      sink += values[(iteration * 37) % channels];
    }
    // Verify a warmed changing-input call and its entire persistent state independently.
    for (int i = 0; i < channels; i++) values[i] = initial[i] + .0078125f;
    System.arraycopy(values, 0, oracleValues, 0, channels);
    System.arraycopy(history, 0, oracleHistory, 0, history.length);
    VectorUtil.causalDepthwiseConv1dSilu(values, 0, history, 0, weights, 0, channels, kernel);
    reference(oracleValues, oracleHistory, weights, lanes, fma);
    int hotMismatches = differences(values, oracleValues) + differences(history, oracleHistory);
    System.out.printf(java.util.Locale.ROOT,
        "{\"channels\":%d,\"kernel\":%d,\"warmup\":%d,\"measured\":%d,\"bits\":%d,\"fma\":%s,\"coldBytes\":%d,\"coldNanos\":%d,\"hotBytesPerCall\":%.3f,\"hotMaxBytes\":%d,\"hotNanosPerCall\":%.3f,\"coldOracleMismatches\":%d,\"hotOracleMismatches\":%d,\"sink\":%.9f,\"outputHash\":%d,\"historyHash\":%d}%n",
        channels, kernel, warm, measured, lanes * 32, fma, coldBytes, coldNanos,
        hotBytes / (double) measured, maxBytes, hotNanos / (double) measured,
        coldMismatches, hotMismatches, sink, java.util.Arrays.hashCode(values), java.util.Arrays.hashCode(history));
  }
  private static int differences(float[] a, float[] b) {
    int count = 0;
    for (int i = 0; i < a.length; i++) if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) count++;
    return count;
  }
  private static void reference(float[] values, float[] history, float[] weights, int lanes, boolean fma) {
    int channels = values.length;
    for (int channel = 0; channel < channels; channel++) {
      float input = values[channel], sum = input * weights[3 * channels + channel];
      for (int tap = 0; tap < 3; tap++) {
        int index = tap * channels + channel;
        float previous = history[index];
        sum = channel < channels - channels % lanes && fma
            ? Math.fma(previous, weights[index], sum) : previous * weights[index] + sum;
        if (tap > 0) history[index - channels] = previous;
      }
      history[2 * channels + channel] = input;
      values[channel] = sum / (1f + (float) StrictMath.exp(-sum));
    }
  }
}
