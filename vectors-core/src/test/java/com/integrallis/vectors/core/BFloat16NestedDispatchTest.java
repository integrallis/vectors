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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BFloat16NestedDispatchTest {
  @Test
  void nestedPlansAndProjectionsCompleteWithoutRepublishingActiveWorkers(@TempDir Path directory)
      throws Exception {
    Path log = directory.resolve("nested-dispatch.log");
    Process process =
        new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-XX:ActiveProcessorCount=4",
                "-cp",
                System.getProperty("java.class.path"),
                BFloat16NestedDispatchProbe.class.getName())
            .redirectErrorStream(true)
            .redirectOutput(log.toFile())
            .start();
    try {
      assertThat(process.waitFor(60, TimeUnit.SECONDS))
          .as("nested dispatch timed out: %s", Files.readString(log))
          .isTrue();
      assertThat(process.exitValue())
          .as("nested dispatch output: %s", Files.readString(log))
          .isZero();
      assertThat(Files.readString(log))
          .contains("CROSS_POOL_DISPATCH_COMPLETE", "NESTED_DISPATCH_COMPLETE");
    } finally {
      if (process.isAlive()) process.destroyForcibly().waitFor();
    }
  }
}
