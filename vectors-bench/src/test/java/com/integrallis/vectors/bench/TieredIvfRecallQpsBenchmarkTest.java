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
package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Smoke test for the tiered IVF recall-vs-QPS bench (P1.3). Runs a tiny configuration end-to-end
 * and asserts that recall increases (or at least does not decrease) with {@code nprobe} — the
 * fundamental contract of an IVF sweep. Without this, a regression that broke routing (e.g., the
 * BuoyIndex skipping clusters) would render the published numbers meaningless even though the bench
 * still printed "successful" rows.
 */
@Tag("unit")
class TieredIvfRecallQpsBenchmarkTest {

  @Test
  void recallIsMonotonicNonDecreasingInNprobe() throws Exception {
    String[] argv = new String[0];
    System.setProperty("bench.tieredIvf.dim", "16");
    System.setProperty("bench.tieredIvf.corpus", "1500");
    System.setProperty("bench.tieredIvf.queries", "20");
    System.setProperty("bench.tieredIvf.nprobe", "1,4,16");

    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    PrintStream priorOut = System.out;
    System.setOut(new PrintStream(captured));
    try {
      TieredIvfRecallQpsBenchmark.main(argv);
    } finally {
      System.setOut(priorOut);
      // Forward the captured output once so a failure dump shows the table.
      priorOut.print(captured.toString());
    }

    String output = captured.toString();
    // Parse the recall column from the sweep table.
    double prev = -1.0;
    int rowsSeen = 0;
    for (String line : output.split("\n")) {
      String trimmed = line.trim();
      if (trimmed.isEmpty() || trimmed.startsWith("nprobe")) continue;
      // Rows look like: "1       0.4067    43647       20.9"
      if (!trimmed.matches("^\\d+\\s+\\d+\\.\\d+\\s+.*$")) continue;
      String[] parts = trimmed.split("\\s+");
      double recall = Double.parseDouble(parts[1]);
      assertThat(recall).as("recall at nprobe=%s must be in [0,1]", parts[0]).isBetween(0.0, 1.0);
      assertThat(recall)
          .as("recall is expected to be monotonically non-decreasing in nprobe")
          .isGreaterThanOrEqualTo(prev);
      prev = recall;
      rowsSeen++;
    }
    assertThat(rowsSeen).as("the sweep table should produce one row per nprobe value").isEqualTo(3);
    assertThat(prev).as("largest-nprobe recall should be > 0").isGreaterThan(0.0);
  }
}
