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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.MemorySegment;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PreparedCosineTest {
  @Test
  void preparedQueryPreservesArrayAndSegmentScoreBits() throws Exception {
    // Vector API reductions can differ between the interpreter and C2. Compare both kernels
    // in the same compilation tier, outside coverage instrumentation, without score tolerances.
    var command = new java.util.ArrayList<String>();
    command.add(java.nio.file.Path.of(System.getProperty("java.home"), "bin", "java").toString());
    command.addAll(
        java.util.List.of(
            "--add-modules=jdk.incubator.vector", "-Xbatch", "-XX:-TieredCompilation"));
    System.getProperties()
        .forEach(
            (key, value) -> {
              if (key.toString().startsWith("vectors.")) command.add("-D" + key + "=" + value);
            });
    String classpath =
        java.nio.file.Path.of(
                VectorUtil.class.getProtectionDomain().getCodeSource().getLocation().toURI())
            + java.io.File.pathSeparator
            + java.nio.file.Path.of(
                PreparedCosineTest.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI());
    command.addAll(java.util.List.of("-cp", classpath, Probe.class.getName()));
    var log = java.nio.file.Files.createTempFile("prepared-cosine-", ".log");
    var process =
        new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
    try {
      org.junit.jupiter.api.Assertions.assertTrue(
          process.waitFor(60, java.util.concurrent.TimeUnit.SECONDS), "probe timeout");
      String output = java.nio.file.Files.readString(log);
      System.out.print(output);
      assertEquals(0, process.exitValue(), output);
    } finally {
      process.destroyForcibly();
      java.nio.file.Files.deleteIfExists(log);
    }
  }

  public static final class Probe {
    private static volatile float sink;

    private static void sameBits(float expected, float actual, String label) {
      if (Float.floatToIntBits(expected) != Float.floatToIntBits(actual)) {
        throw new AssertionError(label + " expected=" + expected + " actual=" + actual);
      }
    }

    public static void main(String[] args) {
      SplittableRandom random = new SplittableRandom(827391);
      int[] dimensions = {
        1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 384, 512, 768, 960, 1536
      };
      for (int dim : dimensions) {
        float[] query = new float[dim], row = new float[dim];
        for (int i = 0; i < dim; i++) {
          query[i] = (float) random.nextDouble(-1, 1);
          row[i] = (float) random.nextDouble(-1, 1);
        }
        MemorySegment q = MemorySegment.ofArray(query), r = MemorySegment.ofArray(row);
        float warmNorm = VectorUtil.cosineQueryNorm(query);
        for (int warm = 0; warm < 20000; warm++) {
          sink = VectorUtil.cosine(query, row);
          sink = VectorUtil.cosineWithQueryNorm(query, row, warmNorm);
          sink = VectorUtil.cosine(q, r, dim);
          sink = VectorUtil.cosineWithQueryNorm(q, r, dim, warmNorm);
        }
        for (int trial = 0; trial < 50; trial++) {
          for (int i = 0; i < dim; i++) row[i] = (float) random.nextDouble(-1, 1);
          float norm = VectorUtil.cosineQueryNorm(query);
          sameBits(
              VectorUtil.cosine(query, row),
              VectorUtil.cosineWithQueryNorm(query, row, norm),
              "array dim=" + dim);
          sameBits(
              VectorUtil.cosine(q, r, dim),
              VectorUtil.cosineWithQueryNorm(q, r, dim, norm),
              "segment dim=" + dim);
        }
        for (float special :
            new float[] {
              0f,
              Float.MIN_VALUE,
              Float.MIN_NORMAL,
              Float.MAX_VALUE,
              Float.POSITIVE_INFINITY,
              Float.NaN
            }) {
          java.util.Arrays.fill(row, special);
          float norm = VectorUtil.cosineQueryNorm(query);
          sameBits(
              VectorUtil.cosine(query, row),
              VectorUtil.cosineWithQueryNorm(query, row, norm),
              "special array dim=" + dim);
          sameBits(
              VectorUtil.cosine(q, r, dim),
              VectorUtil.cosineWithQueryNorm(q, r, dim, norm),
              "special segment dim=" + dim);
          java.util.Arrays.fill(query, special);
          norm = VectorUtil.cosineQueryNorm(query);
          sameBits(
              VectorUtil.cosine(query, row),
              VectorUtil.cosineWithQueryNorm(query, row, norm),
              "special query dim=" + dim);
          sameBits(
              VectorUtil.cosine(q, r, dim),
              VectorUtil.cosineWithQueryNorm(q, r, dim, norm),
              "special segment query dim=" + dim);
        }
      }

      System.out.println("Prepared cosine exact array/segment parity passed");
    }
  }
}
