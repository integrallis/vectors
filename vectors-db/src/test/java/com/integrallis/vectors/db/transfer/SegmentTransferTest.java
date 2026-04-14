package com.integrallis.vectors.db.transfer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SegmentTransferTest {

  @TempDir Path tmp;

  @Test
  void exportThenImport_restoresAllFiles() throws IOException {
    // Create a fake generation directory with several files
    Path src = tmp.resolve("gen-001");
    Files.createDirectories(src);
    Files.write(src.resolve("manifest.bin"), new byte[] {1, 2, 3});
    Files.write(src.resolve("vectors.bin"), new byte[] {4, 5, 6, 7});
    Files.write(src.resolve("idmap.bin"), new byte[] {8, 9});

    // Export
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    SegmentExporter.export(src, bos);

    // Import
    Path dst = tmp.resolve("gen-002");
    SegmentImporter.importTo(new ByteArrayInputStream(bos.toByteArray()), dst);

    // Verify
    assertThat(Files.readAllBytes(dst.resolve("manifest.bin"))).isEqualTo(new byte[] {1, 2, 3});
    assertThat(Files.readAllBytes(dst.resolve("vectors.bin"))).isEqualTo(new byte[] {4, 5, 6, 7});
    assertThat(Files.readAllBytes(dst.resolve("idmap.bin"))).isEqualTo(new byte[] {8, 9});
  }

  @Test
  void importOverwritesExistingFiles() throws IOException {
    Path src = tmp.resolve("src");
    Files.createDirectories(src);
    Files.write(src.resolve("data.bin"), new byte[] {42});
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    SegmentExporter.export(src, bos);

    Path dst = tmp.resolve("dst");
    Files.createDirectories(dst);
    Files.write(dst.resolve("data.bin"), new byte[] {0}); // stale content

    SegmentImporter.importTo(new ByteArrayInputStream(bos.toByteArray()), dst);

    assertThat(Files.readAllBytes(dst.resolve("data.bin"))).isEqualTo(new byte[] {42});
  }

  @Test
  void exportCreatesTargetDirectory() throws IOException {
    Path src = tmp.resolve("gen");
    Files.createDirectories(src);
    Files.write(src.resolve("x.bin"), new byte[] {99});
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    SegmentExporter.export(src, bos);

    Path dst = tmp.resolve("newdir"); // does not exist yet
    SegmentImporter.importTo(new ByteArrayInputStream(bos.toByteArray()), dst);

    assertThat(Files.exists(dst.resolve("x.bin"))).isTrue();
  }

  @Test
  void exportNonExistentDirectoryThrows() {
    assertThatThrownBy(
            () -> SegmentExporter.export(tmp.resolve("missing"), new ByteArrayOutputStream()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void importUnsafeZipEntryThrows() throws IOException {
    // Build a ZIP with a path-traversal entry manually
    Path src = tmp.resolve("src");
    Files.createDirectories(src);
    Files.write(src.resolve("ok.bin"), new byte[] {1});

    // We cannot easily inject "../evil" via SegmentExporter, but we can craft the bytes
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    try (java.util.zip.ZipOutputStream zos = new java.util.zip.ZipOutputStream(bos)) {
      zos.putNextEntry(new java.util.zip.ZipEntry("../evil.bin"));
      zos.write(new byte[] {0});
      zos.closeEntry();
    }

    Path dst = tmp.resolve("safe");
    assertThatThrownBy(
            () -> SegmentImporter.importTo(new ByteArrayInputStream(bos.toByteArray()), dst))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("unsafe");
  }

  @Test
  void largeFileRoundTrip() throws IOException {
    Path src = tmp.resolve("large");
    Files.createDirectories(src);
    byte[] big = new byte[4 * 1024 * 1024]; // 4 MB
    new java.util.Random(42L).nextBytes(big);
    Files.write(src.resolve("vectors.bin"), big);

    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    SegmentExporter.export(src, bos);

    Path dst = tmp.resolve("large-dst");
    SegmentImporter.importTo(new ByteArrayInputStream(bos.toByteArray()), dst);

    assertThat(Files.readAllBytes(dst.resolve("vectors.bin"))).isEqualTo(big);
  }
}
