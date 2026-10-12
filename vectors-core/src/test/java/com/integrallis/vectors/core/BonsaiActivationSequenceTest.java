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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Tag("unit")
class BonsaiActivationSequenceTest {
  @ParameterizedTest(name = "width ceiling={0}, disable FMA={1}")
  @CsvSource({
    "default,false",
    "64,false",
    "128,false",
    "256,false",
    "512,false",
    "default,true",
    "64,true",
    "128,true",
    "256,true",
    "512,true"
  })
  void exactGateAndPersistentHistoryContractsSurviveCompilation(String width, boolean noFma)
      throws Exception {
    Path directory = Path.of("build", "reports", "bonsai-activation-sequence");
    Files.createDirectories(directory);
    Path output = directory.resolve(width + (noFma ? "-no-fma" : "") + ".log");
    List<String> command =
        new ArrayList<>(
            List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Xmx256m"));
    if (!width.equals("default")) command.add("-Dvectors.maxBits=" + width);
    if (noFma)
      command.addAll(List.of("-Dvectors.useVectorFMA=false", "-Dvectors.useScalarFMA=false"));
    command.addAll(
        List.of(
            "-cp",
            System.getProperty("java.class.path"),
            BonsaiActivationSequenceProbe.class.getName()));
    Process process =
        new ProcessBuilder(command)
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
    String text = Files.readString(output);
    assertThat(completed).as("timed out: %s", text).isTrue();
    assertThat(process.exitValue()).as("%s: %s", command, text).isZero();
    assertThat(text).contains("CONFIG requested=", "BONSAI_ACTIVATIONS_STABLE");
  }
}
