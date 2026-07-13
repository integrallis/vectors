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
package com.integrallis.vectors.ingest.sources;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ingest.IngestDoc;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class DirectorySourceTest {

  @Test
  void walksDirectoryEmittingOneDocPerFile(@TempDir Path tmp) throws IOException {
    Files.writeString(tmp.resolve("a.txt"), "alpha", StandardCharsets.UTF_8);
    Files.writeString(tmp.resolve("b.txt"), "beta", StandardCharsets.UTF_8);
    Files.createDirectories(tmp.resolve("sub"));
    Files.writeString(tmp.resolve("sub/c.txt"), "gamma", StandardCharsets.UTF_8);

    DirectorySource src = new DirectorySource("dir", tmp);
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).hasSize(3);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("a.txt", "b.txt", "sub/c.txt");
    assertThat(docs.get(2).text()).isEqualTo("gamma");
    assertThat(src.estimatedSize()).hasValue(3);
  }

  @Test
  void skipsHiddenFiles(@TempDir Path tmp) throws IOException {
    Files.writeString(tmp.resolve(".hidden"), "x", StandardCharsets.UTF_8);
    Files.writeString(tmp.resolve("visible.txt"), "y", StandardCharsets.UTF_8);
    List<IngestDoc> docs = new ArrayList<>();
    new DirectorySource("d", tmp).forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("visible.txt");
  }

  @Test
  void resumesFromOffset(@TempDir Path tmp) throws IOException {
    Files.writeString(tmp.resolve("a.txt"), "1", StandardCharsets.UTF_8);
    Files.writeString(tmp.resolve("b.txt"), "2", StandardCharsets.UTF_8);
    Files.writeString(tmp.resolve("c.txt"), "3", StandardCharsets.UTF_8);
    DirectorySource src = new DirectorySource("d", tmp, 2L, "text/plain", true);
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("c.txt");
  }

  @Test
  void blobModeKeepsBytes(@TempDir Path tmp) throws IOException {
    byte[] payload = {1, 2, 3, 4};
    Files.write(tmp.resolve("a.bin"), payload);
    DirectorySource src = new DirectorySource("d", tmp, 0L, "application/octet-stream", false);
    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).hasSize(1);
    assertThat(docs.get(0).text()).isNull();
    assertThat(docs.get(0).blob()).containsExactly(payload);
    assertThat(docs.get(0).mime()).isEqualTo("application/octet-stream");
  }

  @Test
  void missingDirectoryYieldsEmpty(@TempDir Path tmp) {
    DirectorySource src = new DirectorySource("d", tmp.resolve("nope"));
    assertThat(src.iterator().hasNext()).isFalse();
  }

  @Test
  void fileListIsSnapshotOnFirstAccessNotDoubleWalked(@TempDir Path tmp) throws IOException {
    // Regression (audit ingest #18, double-walk): estimatedSize() and iterator() each used to
    // full-walk+sort the tree. The list is now walked once and reused. Observable via a file added
    // after first access: it must NOT appear (the snapshot is pinned), and both estimatedSize() and
    // iteration reflect the same snapshot.
    Files.writeString(tmp.resolve("a.txt"), "a", StandardCharsets.UTF_8);
    Files.writeString(tmp.resolve("b.txt"), "b", StandardCharsets.UTF_8);
    DirectorySource src = new DirectorySource("d", tmp);

    assertThat(src.estimatedSize()).hasValue(2); // first walk snapshots {a, b}

    Files.writeString(tmp.resolve("c.txt"), "c", StandardCharsets.UTF_8); // added after the snapshot
    assertThat(src.estimatedSize()).as("snapshot is reused, not re-walked").hasValue(2);

    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);
    assertThat(docs).extracting(IngestDoc::id).containsExactly("a.txt", "b.txt");
  }

  @Test
  void rejectsFilesLargerThanTheCap(@TempDir Path tmp) throws IOException {
    // Regression (audit ingest #18): next() used to readAllBytes() unconditionally, so a single
    // oversized file OOM'd the heap. It now stats the file first and fails fast past the cap.
    Files.writeString(tmp.resolve("small.txt"), "ok", StandardCharsets.UTF_8); // 2 bytes
    Files.write(tmp.resolve("big.bin"), new byte[64]); // 64 bytes > 16-byte cap
    DirectorySource src = new DirectorySource("d", tmp, 0L, "application/octet-stream", false, 16L);

    assertThatThrownBy(() -> src.forEach(d -> {}))
        .isInstanceOf(UncheckedIOException.class)
        .hasMessageContaining("per-file cap")
        .hasMessageContaining("big.bin");
  }

  @Test
  void filesAtOrUnderCapAreRead(@TempDir Path tmp) throws IOException {
    Files.write(tmp.resolve("a.bin"), new byte[16]); // exactly at the cap → allowed
    DirectorySource src = new DirectorySource("d", tmp, 0L, "application/octet-stream", false, 16L);

    List<IngestDoc> docs = new ArrayList<>();
    src.forEach(docs::add);

    assertThat(docs).hasSize(1);
    assertThat(docs.get(0).blob()).hasSize(16);
  }

  @Test
  void defaultCapMatchesTurbopufferDocumentLimit() {
    assertThat(DirectorySource.DEFAULT_MAX_FILE_BYTES).isEqualTo(64L * 1024 * 1024);
  }

  @Test
  void rejectsNonPositiveCap(@TempDir Path tmp) {
    assertThatThrownBy(() -> new DirectorySource("d", tmp, 0L, "text/plain", true, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxFileBytes");
  }
}
