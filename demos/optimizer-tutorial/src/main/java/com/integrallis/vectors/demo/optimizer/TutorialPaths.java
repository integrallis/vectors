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
package com.integrallis.vectors.demo.optimizer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Shared paths and JSON helpers used by the four tutorial stages. */
final class TutorialPaths {

  /** Snapshot of a single configuration's measurements; serialised between stages. */
  record Snapshot(
      String label,
      String datasetName,
      int corpusSize,
      int queryCount,
      int kForMetrics,
      int m,
      int efConstruction,
      int efSearch,
      double recallAtK,
      double ndcgAtK,
      double p50LatencyUs,
      double p95LatencyUs,
      double p99LatencyUs,
      long buildTimeMs,
      double objectiveScore) {}

  private static final ObjectMapper MAPPER =
      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

  private TutorialPaths() {}

  /** Tutorial root directory: {@code ~/.vectors/optimizer/tutorial}. */
  static Path root() {
    Path p = Path.of(System.getProperty("user.home", "."), ".vectors", "optimizer", "tutorial");
    try {
      Files.createDirectories(p);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to create " + p, ioe);
    }
    return p;
  }

  static Path baselineFile() {
    return root().resolve("baseline.json");
  }

  static Path bestFile() {
    return root().resolve("best.json");
  }

  static void writeJson(Path file, Object value) {
    try {
      MAPPER.writeValue(file.toFile(), value);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to write " + file, ioe);
    }
  }

  static Snapshot readSnapshot(Path file) {
    if (!Files.exists(file)) return null;
    try {
      return MAPPER.readValue(file.toFile(), Snapshot.class);
    } catch (IOException ioe) {
      throw new UncheckedIOException("Failed to read " + file, ioe);
    }
  }
}
