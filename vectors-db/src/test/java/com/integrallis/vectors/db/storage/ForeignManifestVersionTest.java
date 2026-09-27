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
package com.integrallis.vectors.db.storage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A collection from an older format must say so, rather than blaming the filesystem.
 *
 * <p>Written because the 0.1.23 to 0.1.24 upgrade did the opposite. {@code tryReadManifest}
 * collapses every read failure to {@code null} — correct for corruption, since falling back to an
 * earlier generation is the right recovery — but a version mismatch affects every generation
 * equally. Recovery therefore found nothing valid, bootstrapped a fresh {@code gen-0}, and failed
 * with "generation directory already exists": a message pointing at directory state rather than at
 * the format. Anyone upgrading would investigate permissions and stale temp directories before
 * suspecting a version.
 */
class ForeignManifestVersionTest {

  /**
   * Writes a manifest header carrying {@code version}, with correct magic and nothing else valid.
   */
  private static void writeManifestWithVersion(Path genDir, int version, int headerLength)
      throws IOException {
    Files.createDirectories(genDir);
    byte[] header = new byte[headerLength];
    ByteBuffer buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(FileFormat.MAGIC_MANIFEST);
    buffer.putInt(version);
    buffer.putInt(headerLength);
    Files.write(genDir.resolve(FileFormat.MANIFEST_FILE), header);
  }

  @Test
  @DisplayName("peekManifestVersion reads the version without validating the rest")
  void peekReadsTheVersionAlone() throws IOException {
    Path gen = Files.createTempDirectory("gen");
    // Deliberately a version-4, 164-byte header: exactly what a 0.1.23 collection carries, and what
    // the bundled router index in models turned out to be.
    writeManifestWithVersion(gen, 4, 164);
    assertEquals(4, GenerationDirectory.peekManifestVersion(gen));
  }

  @Test
  @DisplayName("a file that is not a manifest peeks as null rather than a bogus version")
  void nonManifestPeeksNull(@TempDir Path root) throws IOException {
    Path gen = root.resolve("gen-0000000000000000");
    Files.createDirectories(gen);
    Files.write(gen.resolve(FileFormat.MANIFEST_FILE), "not a manifest at all".getBytes());
    assertNull(GenerationDirectory.peekManifestVersion(gen));
  }

  @Test
  @DisplayName("a truncated manifest peeks as null, not as a partially-read version")
  void truncatedPeeksNull(@TempDir Path root) throws IOException {
    Path gen = root.resolve("gen-0000000000000000");
    Files.createDirectories(gen);
    Files.write(gen.resolve(FileFormat.MANIFEST_FILE), new byte[] {1, 2, 3});
    assertNull(GenerationDirectory.peekManifestVersion(gen));
  }

  @Test
  @DisplayName("an absent manifest peeks as null")
  void absentPeeksNull(@TempDir Path root) {
    assertNull(GenerationDirectory.peekManifestVersion(root.resolve("gen-0000000000000000")));
  }

  @Test
  @DisplayName(
      "the current version peeks as itself, so the check cannot misfire on a good collection")
  void currentVersionPeeksAsItself(@TempDir Path root) throws IOException {
    Path gen = root.resolve("gen-0000000000000000");
    writeManifestWithVersion(gen, FileFormat.VERSION_MANIFEST, Manifest.HEADER_SIZE);
    assertEquals(FileFormat.VERSION_MANIFEST, GenerationDirectory.peekManifestVersion(gen));
  }

  @Test
  @DisplayName("recovering an older collection names the version and says to rebuild")
  void recoveringAForeignVersionExplainsItself(@TempDir Path root) throws IOException {
    // The exact shape of the failure that started this: a version-4 generation directory present,
    // and
    // recovery about to bootstrap over it.
    writeManifestWithVersion(root.resolve("gen-0000000000000000"), 4, 164);
    Files.writeString(root.resolve(FileFormat.CURRENT_FILE), "0");

    IOException failure =
        org.junit.jupiter.api.Assertions.assertThrows(
            IOException.class, () -> GenerationDirectory.recover(root, null, null));

    String message = failure.getMessage();
    assertTrue(message.contains("version 4"), message);
    assertTrue(
        message.contains("version " + FileFormat.VERSION_MANIFEST),
        "the message must name the version this build reads: " + message);
    assertTrue(message.contains("rebuild"), "it must say what to do: " + message);
    assertTrue(
        !message.contains("already exists"),
        "it must NOT blame directory state, which is what it used to do: " + message);
  }
}
