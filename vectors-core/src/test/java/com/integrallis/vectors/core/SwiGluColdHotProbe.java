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

import java.util.ArrayList;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;

/** Public SwiGLU must preserve numerical accuracy and exact cold/hot, alias and chunk results. */
final class SwiGluColdHotProbe {
  private static final int LENGTH = 259;
  private static final int OFFSET = 3;
  private static final float[] EDGES = {
    0.0f,
    -0.0f,
    Float.MIN_VALUE,
    -Float.MIN_VALUE,
    Float.MIN_NORMAL,
    -Float.MIN_NORMAL,
    Float.MAX_VALUE,
    -Float.MAX_VALUE,
    88.72283f,
    88.72284f,
    -88.72283f,
    -88.72284f,
    103.97207f,
    103.97208f,
    -103.97207f,
    -103.97208f,
    1.0f,
    -1.0f,
    20.0f,
    -20.0f,
    100.0f,
    -100.0f
  };

  public static void main(String[] args) throws Exception {
    check(4021);
    try (var executor = Executors.newFixedThreadPool(4)) {
      var work = new ArrayList<Callable<Void>>();
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
    System.out.println("SWIGLU_STABLE exact cold/hot, chunks, offsets, tails, aliases, concurrent");
  }

  private static void check(long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[] gate = new float[LENGTH + OFFSET * 2];
    float[] up = new float[gate.length];
    for (int index = 0; index < gate.length; index++) {
      gate[index] = index < EDGES.length ? EDGES[index] : (float) random.nextDouble(-100, 100);
      up[index] = (float) random.nextDouble(-1, 1);
    }
    System.arraycopy(EDGES, 0, gate, OFFSET, EDGES.length);
    float[] expected = new float[LENGTH];
    float[] allAliasedExpected = new float[LENGTH];
    float[] cold = new float[gate.length];
    float[] coldAll = new float[gate.length];
    VectorUtil.swiGlu(cold, OFFSET, gate, OFFSET, up, OFFSET, LENGTH);
    VectorUtil.swiGlu(coldAll, OFFSET, gate, OFFSET, gate, OFFSET, LENGTH);
    for (int index = 0; index < LENGTH; index++) {
      float value = gate[OFFSET + index];
      float activated = value / (1.0f + (float) StrictMath.exp(-value));
      expected[index] = cold[OFFSET + index];
      allAliasedExpected[index] = coldAll[OFFSET + index];
      accuracy(expected[index], activated * up[OFFSET + index], up[OFFSET + index]);
      accuracy(allAliasedExpected[index], activated * value, value);
    }
    float[] out = new float[gate.length];
    float[] gateAlias = new float[gate.length];
    float[] upAlias = new float[gate.length];
    float[] allAlias = new float[gate.length];
    float[] split = new float[gate.length];
    int[] chunks = {1, 7, 16, 33};
    for (int iteration = 0; iteration < 5000; iteration++) {
      System.arraycopy(gate, 0, out, 0, gate.length);
      System.arraycopy(gate, 0, gateAlias, 0, gate.length);
      System.arraycopy(up, 0, upAlias, 0, up.length);
      System.arraycopy(gate, 0, allAlias, 0, gate.length);
      System.arraycopy(gate, 0, split, 0, gate.length);
      VectorUtil.swiGlu(out, OFFSET, gate, OFFSET, up, OFFSET, LENGTH);
      VectorUtil.swiGlu(gateAlias, OFFSET, gateAlias, OFFSET, up, OFFSET, LENGTH);
      VectorUtil.swiGlu(upAlias, OFFSET, gate, OFFSET, upAlias, OFFSET, LENGTH);
      VectorUtil.swiGlu(allAlias, OFFSET, allAlias, OFFSET, allAlias, OFFSET, LENGTH);
      int chunk = chunks[iteration % chunks.length];
      for (int start = 0; start < LENGTH; start += chunk) {
        int offset = OFFSET + start;
        VectorUtil.swiGlu(split, offset, gate, offset, up, offset, Math.min(chunk, LENGTH - start));
      }
      verify(out, gate, expected, seed, iteration, "separate");
      verify(gateAlias, gate, expected, seed, iteration, "gate alias");
      verify(upAlias, up, expected, seed, iteration, "up alias");
      verify(allAlias, gate, allAliasedExpected, seed, iteration, "all alias");
      verify(split, gate, expected, seed, iteration, "chunk " + chunk);
    }
  }

  private static void accuracy(float actual, float reference, float up) {
    if (!Float.isFinite(reference) || reference == 0f) {
      if (Float.floatToIntBits(actual) != Float.floatToIntBits(reference))
        throw new AssertionError("special-value mismatch");
    } else if (Math.abs((double) actual - reference) > 2e-6 * Math.abs(up) + Math.ulp(reference)) {
      throw new AssertionError("published numerical accuracy exceeded");
    }
  }

  private static void verify(
      float[] actual, float[] guards, float[] expected, long seed, int iteration, String mode) {
    for (int index = 0; index < actual.length; index++) {
      float value =
          index >= OFFSET && index < OFFSET + LENGTH ? expected[index - OFFSET] : guards[index];
      if (Float.floatToRawIntBits(actual[index]) != Float.floatToRawIntBits(value)) {
        throw new AssertionError(
            "SwiGLU mismatch seed="
                + seed
                + " iteration="
                + iteration
                + " mode="
                + mode
                + " cell="
                + index
                + " actual="
                + actual[index]
                + " expected="
                + value);
      }
    }
  }
}
