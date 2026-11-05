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

/**
 * Parity test for the Q4_K batch-scale-unpack optimization.
 *
 * <p>The pure-Java Q4_K path is currently "dead code" from a product perspective — native Rust
 * kernels handle Q4_K in production. This test ensures the pure-Java path remains correct and
 * bit-identical, so it can be reactivated when the JVM Vector API catches up.
 *
 * <p>Runs the same Q4_K cold/hot determinism probe twice:
 *
 * <ol>
 *   <li>With batch unpack enabled (default optimized path)
 *   <li>With batch unpack disabled (legacy scalar path)
 * </ol>
 *
 * Both must pass with exit code 0, proving parity.
 */
@Tag("unit")
class Q4KBatchUnpackParityTest {

  @Test
  void batchUnpackIsBitIdenticalToScalarPath256Bit() throws Exception {
    runBatchParityProbe(256);
  }

  @Test
  void batchUnpackIsBitIdenticalToScalarPath128Bit() throws Exception {
    runBatchParityProbe(128);
  }

  private static void runBatchParityProbe(int maxBits) throws Exception {
    // Run with batch unpack ENABLED (default optimized path)
    Process batchProcess =
        ForkedJvmCoverage.process(
                javaExecutable(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Dvectors.maxBits=" + maxBits,
                "-Dvectors.gguf.parallel=false",
                "-Dvectors.gguf.q4k.useBatchUnpack=true",
                "-cp",
                System.getProperty("java.class.path"),
                Q4KColdHotDeterminismProbe.class.getName())
            .redirectErrorStream(true)
            .start();

    boolean batchCompleted = batchProcess.waitFor(60, TimeUnit.SECONDS);
    String batchOutput = readOutput(batchProcess);
    assertThat(batchCompleted).as("batch probe timed out; output:%n%s", batchOutput).isTrue();
    assertThat(batchProcess.exitValue()).as("batch probe output:%n%s", batchOutput).isZero();

    // Run with batch unpack DISABLED (legacy scalar path)
    Process scalarProcess =
        ForkedJvmCoverage.process(
                javaExecutable(),
                "--add-modules",
                "jdk.incubator.vector",
                "-Dvectors.maxBits=" + maxBits,
                "-Dvectors.gguf.parallel=false",
                "-Dvectors.gguf.q4k.useBatchUnpack=false",
                "-cp",
                System.getProperty("java.class.path"),
                Q4KColdHotDeterminismProbe.class.getName())
            .redirectErrorStream(true)
            .start();

    boolean scalarCompleted = scalarProcess.waitFor(60, TimeUnit.SECONDS);
    String scalarOutput = readOutput(scalarProcess);
    assertThat(scalarCompleted).as("scalar probe timed out; output:%n%s", scalarOutput).isTrue();
    assertThat(scalarProcess.exitValue()).as("scalar probe output:%n%s", scalarOutput).isZero();
  }

  private static String javaExecutable() {
    return Path.of(System.getProperty("java.home"), "bin", "java").toString();
  }

  private static String readOutput(Process process) throws IOException {
    return new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
  }
}
