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

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SwiGluColdHotTest {
  @TempDir Path temporary;

  @Test
  void swiGluPreservesExactContractAcrossCompilationAndConcurrentCallers() throws Exception {
    assumeTrue(
        publishedActivationRouteAvailable(),
        "activations only carry the cold/hot contract on the published aarch64 route");
    for (int bits : new int[] {128}) {
      Path output = temporary.resolve("swiglu-" + bits + ".log");
      Process process =
          ForkedJvmCoverage.process(
                  Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                  "--add-modules",
                  "jdk.incubator.vector",
                  "-Dvectors.maxBits=" + bits,
                  "-cp",
                  System.getProperty("java.class.path"),
                  SwiGluColdHotProbe.class.getName())
              .redirectErrorStream(true)
              .redirectOutput(output.toFile())
              .start();
      boolean completed;
      try {
        completed = process.waitFor(90, TimeUnit.SECONDS);
      } finally {
        if (process.isAlive()) {
          process.destroyForcibly();
          process.waitFor();
        }
      }
      String text = java.nio.file.Files.readString(output);
      assertThat(completed).as("probe timed out: %s", text).isTrue();
      assertThat(process.exitValue()).as("probe failed: %s", text).isZero();
      assertThat(text).contains("SWIGLU_STABLE");
    }
  }

  // Same scope as ActivationColdHotTest and BonsaiActivationSequenceTest, for the same reason.
  // Activations route to PublishedArmActivations only when USE_PUBLISHED_ARM_ACTIVATIONS holds,
  // which needs aarch64 AND a 128-bit float species; every other configuration runs the released
  // vector EXP intrinsic, whose rounding moves when HotSpot intrinsifies it. See RELEASE-TODO.md
  // for why the deterministic substitute was not adopted.
  private static boolean publishedActivationRouteAvailable() {
    return "aarch64".equals(System.getProperty("os.arch"));
  }
}
