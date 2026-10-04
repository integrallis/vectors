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

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class OffsetBatchCosineBitsTest {
  @Test
  void offsetBatchPreservesWarmedScoreBits() throws Exception {
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
                OffsetBatchCosineBitsTest.class
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

    public static void main(String[] args) {
      var random = new SplittableRandom(17273);
      for (int dim : new int[] {1, 7, 16, 31, 32, 65, 100, 128, 129, 512, 768}) {
        float[] query = new float[dim];
        float[][] rows = new float[9][dim];
        MemorySegment[] segments = new MemorySegment[9];
        MemorySegment matrix = MemorySegment.ofArray(new float[9 * (dim + 13)]);
        long[] offsets = new long[9];
        for (int d = 0; d < dim; d++) query[d] = (float) random.nextDouble(-1, 1);
        for (int r = 0; r < rows.length; r++) {
          for (int d = 0; d < dim; d++) rows[r][d] = (float) random.nextDouble(-1, 1);
          offsets[r] = (long) r * (dim + 13) * Float.BYTES;
          segments[r] = matrix.asSlice(offsets[r], (long) dim * Float.BYTES);
          MemorySegment.copy(rows[r], 0, segments[r], java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, dim);
        }
        float[] expected = new float[9], actual = new float[9];
        for (int warm = 0; warm < 12000; warm++) {
          float norm = VectorUtil.batchCosineQueryNorm(query);
          VectorUtil.batchCosine(query, segments, dim, expected, 9);
          VectorUtil.batchCosineWithQueryNorm(query, matrix, offsets, dim, norm, actual, 9);
          sink = expected[0] + actual[0];
        }
        for (int count = 0; count <= 9; count++) {
          float norm = VectorUtil.batchCosineQueryNorm(query);
          VectorUtil.batchCosine(query, segments, dim, expected, count);
          VectorUtil.batchCosineWithQueryNorm(query, matrix, offsets, dim, norm, actual, count);
          bits(expected, actual, count);
        }
      }
    }

    private static void bits(float[] a, float[] b, int count) {
      for (int i = 0; i < count; i++)
        if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i]))
          throw new AssertionError("score bits differ at " + i);
    }
  }
}
