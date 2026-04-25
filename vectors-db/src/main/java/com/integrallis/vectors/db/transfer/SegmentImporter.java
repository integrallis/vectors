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
package com.integrallis.vectors.db.transfer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Imports a generation archive produced by {@link SegmentExporter} into a target directory.
 *
 * <p>Each ZIP entry is written to {@code targetDir/<entryName>}. Existing files are overwritten.
 * The caller is responsible for ensuring {@code targetDir} does not contain a live generation that
 * is concurrently accessed.
 */
public final class SegmentImporter {

  private SegmentImporter() {}

  /**
   * Extracts all entries from the ZIP-formatted {@code in} into {@code targetDir}, creating the
   * directory if it does not exist.
   *
   * @param in source stream (caller-owned; caller closes)
   * @param targetDir destination directory
   * @throws IOException on I/O failure or if a ZIP entry tries to escape the target directory
   *     (path-traversal guard)
   */
  public static void importTo(InputStream in, Path targetDir) throws IOException {
    Files.createDirectories(targetDir);
    try (ZipInputStream zis = new ZipInputStream(in)) {
      ZipEntry entry;
      while ((entry = zis.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          zis.closeEntry();
          continue;
        }
        String name = entry.getName();
        // Path-traversal guard: reject any entry whose name contains ".."
        if (name.contains("..") || name.contains("/") || name.contains("\\")) {
          throw new IOException("refusing unsafe ZIP entry name: " + name);
        }
        Path target = targetDir.resolve(name);
        Files.copy(zis, target, StandardCopyOption.REPLACE_EXISTING);
        zis.closeEntry();
      }
    }
  }
}
