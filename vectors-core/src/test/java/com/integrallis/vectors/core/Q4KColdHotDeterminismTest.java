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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Q4KColdHotDeterminismTest {

  @Test
  void q4_K256BitKernelIsBitIdenticalBeforeAndAfterCompilation() throws Exception {
    Process process =
        new ProcessBuilder(
                javaExecutable(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Dvectors.maxBits=256",
                "-Dvectors.gguf.parallel=false",
                "-cp",
                System.getProperty("java.class.path"),
                Q4KColdHotDeterminismProbe.class.getName())
            .redirectErrorStream(true)
            .start();

    boolean completed = process.waitFor(60, TimeUnit.SECONDS);
    String output = readOutput(process);

    assertThat(completed).as("probe timed out; output:%n%s", output).isTrue();
    assertThat(process.exitValue()).as("probe output:%n%s", output).isZero();
  }

  private static String javaExecutable() {
    return Path.of(System.getProperty("java.home"), "bin", "java").toString();
  }

  private static String readOutput(Process process) throws IOException {
    return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
  }
}
