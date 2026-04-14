package com.integrallis.vectors.db.transfer;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Exports a generation directory to a ZIP-formatted byte stream. The exported archive can be
 * transferred to a peer node and restored via {@link SegmentImporter}.
 *
 * <p>All regular files directly inside the {@code generationDir} are included. Sub-directories are
 * not recursed — a generation directory is a flat set of binary segment files.
 */
public final class SegmentExporter {

  private SegmentExporter() {}

  /**
   * Writes a ZIP archive of all files in {@code generationDir} to {@code out}. The stream is
   * flushed but not closed.
   *
   * @param generationDir source directory (must exist and contain at least one file)
   * @param out destination stream (caller-owned)
   * @throws IOException on I/O failure
   */
  public static void export(Path generationDir, OutputStream out) throws IOException {
    if (!Files.isDirectory(generationDir)) {
      throw new IllegalArgumentException("not a directory: " + generationDir);
    }
    try (ZipOutputStream zos = new ZipOutputStream(out)) {
      try (var stream = Files.list(generationDir)) {
        for (Path file : (Iterable<Path>) stream.filter(Files::isRegularFile)::iterator) {
          ZipEntry entry = new ZipEntry(file.getFileName().toString());
          entry.setSize(Files.size(file));
          zos.putNextEntry(entry);
          Files.copy(file, zos);
          zos.closeEntry();
        }
      }
      zos.finish();
    }
  }
}
