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
package com.integrallis.vectors.studio.sidecart.sources;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FileSidecartSourceTest {

  @Test
  void readsTextFileForId(@TempDir Path tmp) throws Exception {
    Files.writeString(tmp.resolve("doc-1.txt"), "hello world");
    FileSidecartSource src = new FileSidecartSource(tmp, ".txt", true, "text/plain");

    Optional<SidecartRecord> rec = src.get("doc-1");

    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isEqualTo("hello world");
    assertThat(rec.get().blob()).isNull();
    assertThat(rec.get().mime()).isEqualTo("text/plain");
  }

  @Test
  void readsBinaryFileForId(@TempDir Path tmp) throws Exception {
    byte[] payload = {1, 2, 3, 4, 5};
    Files.write(tmp.resolve("img-7.png"), payload);
    FileSidecartSource src = new FileSidecartSource(tmp, ".png", false, "image/png");

    Optional<SidecartRecord> rec = src.get("img-7");

    assertThat(rec).isPresent();
    assertThat(rec.get().blob()).isEqualTo(payload);
    assertThat(rec.get().text()).isNull();
    assertThat(rec.get().mime()).isEqualTo("image/png");
  }

  @Test
  void returnsEmptyForUnknownId(@TempDir Path tmp) {
    FileSidecartSource src = new FileSidecartSource(tmp, ".png", false, "image/png");
    assertThat(src.get("nope")).isEmpty();
  }

  @Test
  void traversalAttemptsResolveInsideBaseDirAndReturnEmpty(@TempDir Path tmp) throws Exception {
    // sanitise() rewrites '/' and '\' to '_' so the path can never leave baseDir; an attempted
    // traversal collapses to a sanitised filename that simply does not exist.
    Files.writeString(tmp.resolve("x.txt"), "x");
    FileSidecartSource src = new FileSidecartSource(tmp, ".txt", true, "text/plain");

    assertThat(src.get("../etc/passwd")).isEmpty();
  }

  @Test
  void sanitisesIdsWithSeparators(@TempDir Path tmp) throws Exception {
    // id 'pdf:p1' should resolve as file 'pdf_p1.txt'
    Files.write(tmp.resolve("pdf_p1.txt"), "page 1".getBytes(StandardCharsets.UTF_8));
    FileSidecartSource src = new FileSidecartSource(tmp, ".txt", true, "text/plain");
    Optional<SidecartRecord> rec = src.get("pdf:p1");
    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isEqualTo("page 1");
  }
}
