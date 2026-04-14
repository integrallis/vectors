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
