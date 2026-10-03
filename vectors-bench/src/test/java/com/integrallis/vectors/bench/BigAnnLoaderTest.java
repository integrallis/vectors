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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.bench.dataset.BigAnnLoader;
import com.integrallis.vectors.bench.dataset.DatasetRegistry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
@DisplayName("BigAnnLoader")
class BigAnnLoaderTest {

  @Nested
  @DisplayName("synthetic files")
  class Synthetic {

    @Test
    @DisplayName("round-trips a hand-written vector file")
    void roundTripsVectors(@TempDir Path dir) throws IOException {
      Path file = dir.resolve("vectors.bin");
      float[][] expected = {{1f, 2f, 3f}, {-4f, 5.5f, 6f}};
      writeVectors(file, expected);

      assertThat(BigAnnLoader.readShape(file).numPoints()).isEqualTo(2);
      assertThat(BigAnnLoader.readShape(file).dimensions()).isEqualTo(3);
      assertThat(BigAnnLoader.readVectors(file)).isDeepEqualTo(expected);
    }

    @Test
    @DisplayName("fails loudly when the header disagrees with the file length")
    void rejectsHeaderLengthMismatch(@TempDir Path dir) throws IOException {
      // This is exactly what an un-rewritten cropped download looks like: a header claiming the
      // full corpus over a file holding a prefix. Reading it as declared would run off the end, and
      // silently returning a short or zero-padded array would corrupt every recall number derived
      // from it.
      Path file = dir.resolve("cropped.bin");
      writeVectors(file, new float[][] {{1f, 2f, 3f}});
      ByteBuffer header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(99);
      try (var channel = Files.newByteChannel(file, java.nio.file.StandardOpenOption.WRITE)) {
        channel.position(0);
        channel.write(header.rewind());
      }

      assertThatThrownBy(() -> BigAnnLoader.readVectors(file))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("point count rewritten");
    }

    @Test
    @DisplayName("reads a prefix when given an explicit count")
    void readsExplicitPrefix(@TempDir Path dir) throws IOException {
      Path file = dir.resolve("vectors.bin");
      writeVectors(file, new float[][] {{1f, 2f}, {3f, 4f}, {5f, 6f}});
      assertThat(BigAnnLoader.readVectors(file, 2, 2))
          .isDeepEqualTo(new float[][] {{1f, 2f}, {3f, 4f}});
    }

    @Test
    void rejectsTruncatedPrefixAndWrongDimension(@TempDir Path dir) throws IOException {
      Path file = dir.resolve("short.bin");
      writeVectors(file, new float[][] {{1f, 2f}});
      assertThatThrownBy(() -> BigAnnLoader.readVectors(file, 2, 2))
          .isInstanceOf(IllegalStateException.class);
      assertThatThrownBy(() -> BigAnnLoader.readVectors(file, 1, 1))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsTruncatedGroundTruthIds(@TempDir Path dir) throws IOException {
      Path file = dir.resolve("short.gt");
      ByteBuffer bytes = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
      bytes.putInt(2).putInt(1).putInt(17); // second query's neighbor is missing
      Files.write(file, bytes.array());
      assertThatThrownBy(() -> BigAnnLoader.readGroundTruth(file))
          .isInstanceOf(java.io.UncheckedIOException.class);
    }

    private static void writeVectors(Path file, float[][] vectors) throws IOException {
      int dims = vectors[0].length;
      ByteBuffer buffer =
          ByteBuffer.allocate(8 + vectors.length * dims * Float.BYTES)
              .order(ByteOrder.LITTLE_ENDIAN);
      buffer.putInt(vectors.length).putInt(dims);
      for (float[] v : vectors) {
        for (float f : v) {
          buffer.putFloat(f);
        }
      }
      Files.write(file, buffer.array());
    }
  }

  @Nested
  @DisplayName("published Cohere Wikipedia files")
  @EnabledIf(
      "com.integrallis.vectors.bench.dataset.DatasetRegistry#wikipediaCohereQueriesAvailable")
  class PublishedFiles {

    @Test
    @DisplayName("reads the 5,000 x 768 query set")
    void readsQueries() {
      float[][] queries = BigAnnLoader.readVectors(DatasetRegistry.wikipediaCohereQueries());
      assertThat(queries).hasDimensions(5_000, 768);
      // Pins the metric choice. These vectors are NOT unit-normalized -- measured query norms run
      // 0.74-0.91 while base norms run 12.5-15.2 -- so cosine and inner product rank differently
      // and
      // only inner product matches the published ground truth. If a future rebuild of this dataset
      // ships normalized vectors, this assertion fails and the metric must be revisited rather than
      // silently scoring against the wrong truth.
      double norm = 0;
      for (float f : queries[0]) {
        norm += (double) f * f;
      }
      assertThat(Math.sqrt(norm)).isBetween(0.7, 0.95);
    }

    @Test
    @DisplayName("reads the published ground truth as 5,000 x 100 ids")
    void readsGroundTruth() {
      int[][] truth = BigAnnLoader.readGroundTruth(DatasetRegistry.wikipediaCohereGroundTruth1M());
      assertThat(truth).hasDimensions(5_000, 100);
      // Ground truth for the 1M crop must only reference ordinals inside that crop.
      for (int[] row : truth) {
        for (int id : row) {
          assertThat(id).isBetween(0, 999_999);
        }
      }
    }
  }
}
