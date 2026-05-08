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

import com.integrallis.vectors.ingest.IngestDoc;
import java.io.IOException;
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
}
