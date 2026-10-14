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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Tag("unit")
class BFloat16MatrixColdHotTest {
  @ParameterizedTest
  @CsvSource({
    "64,48,5120",
    "128,48,5120",
    "256,48,5120",
    "64,5,37",
    "128,5,37",
    "256,5,37",
    "64,8193,37",
    "128,8193,37",
    "256,8193,37",
    "64,512,2048",
    "128,512,2048",
    "256,512,2048"
  })
  void projectionsAreBitIdenticalAcrossCompilationAndBatching(
      int maxBits, int rows, int columns, @TempDir Path directory) throws Exception {
    Path log = directory.resolve("probe.log");
    Process process =
        ForkedJvmCoverage.process(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-XX:ActiveProcessorCount=4",
                "-Xmx512m",
                "-Dvectors.maxBits=" + maxBits,
                "-cp",
                System.getProperty("java.class.path"),
                BFloat16MatrixColdHotProbe.class.getName(),
                Integer.toString(rows),
                Integer.toString(columns))
            .redirectErrorStream(true)
            .redirectOutput(log.toFile())
            .start();
    try {
      boolean completed = process.waitFor(180, TimeUnit.SECONDS);
      assertThat(completed).as("probe timed out; output:%n%s", Files.readString(log)).isTrue();
      assertThat(process.exitValue()).as("probe output:%n%s", Files.readString(log)).isZero();
    } finally {
      if (process.isAlive()) process.destroyForcibly().waitFor();
    }
  }
}
