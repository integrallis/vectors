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
package com.integrallis.vectors.storage.backend;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The backend is a filesystem sandbox: no key, however crafted, may read or write outside its root.
 * Real directories, real symlinks, no doubles.
 */
@Tag("unit")
class LocalFileStorageBackendSandboxTest {

  @TempDir Path tmp;

  private Path root() throws IOException {
    return Files.createDirectories(tmp.resolve("root"));
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "../escape",
        "a/../../escape",
        "a/b/../../../escape",
        "..",
        "./../escape",
      })
  void dotDotSegmentsAreRejected(String key) throws IOException {
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root());
    assertThatThrownBy(() -> backend.put(key, new byte[] {1}))
        .isInstanceOf(IOException.class)
        .hasMessageContaining(key);
    assertThat(Files.exists(tmp.resolve("escape"))).isFalse();
  }

  @Test
  void absoluteKeysAreRejected() throws IOException {
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root());
    Path outside = tmp.resolve("outside.bin");
    assertThatThrownBy(() -> backend.put(outside.toString(), new byte[] {1}))
        .isInstanceOf(IOException.class);
    assertThat(Files.exists(outside)).isFalse();
  }

  @ParameterizedTest
  @ValueSource(strings = {"", "/", ".", "a/", "a//b", "nul\u0000byte"})
  void emptyDotAndMalformedKeysAreRejectedAsIOException(String key) throws IOException {
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root());
    assertThatThrownBy(() -> backend.put(key, new byte[] {1})).isInstanceOf(IOException.class);
    assertThatThrownBy(() -> backend.get(key)).isInstanceOf(IOException.class);
  }

  @Test
  void aSymlinkInsideTheRootCannotBeUsedToWriteOutsideIt() throws IOException {
    Path root = root();
    Path outside = Files.createDirectories(tmp.resolve("outside"));
    Files.createSymbolicLink(root.resolve("link"), outside);
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root);

    assertThatThrownBy(() -> backend.put("link/escaped.bin", new byte[] {1}))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("outside");
    assertThat(Files.exists(outside.resolve("escaped.bin"))).isFalse();
    try (Stream<Path> s = Files.list(outside)) {
      assertThat(s).isEmpty();
    }
  }

  @Test
  void aSymlinkInsideTheRootCannotBeUsedToReadOutsideIt() throws IOException {
    Path root = root();
    Path outside = Files.createDirectories(tmp.resolve("outside"));
    Files.write(outside.resolve("secret.bin"), new byte[] {42});
    Files.createSymbolicLink(root.resolve("link"), outside);
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root);

    assertThatThrownBy(() -> backend.get("link/secret.bin")).isInstanceOf(IOException.class);
    assertThatThrownBy(() -> backend.getRange("link/secret.bin", 0, 1))
        .isInstanceOf(IOException.class);
    assertThatThrownBy(() -> backend.open("link/secret.bin")).isInstanceOf(IOException.class);
  }

  @Test
  void aSymlinkedRootIsResolvedToItsRealDirectory() throws IOException {
    Path real = Files.createDirectories(tmp.resolve("real-root"));
    Path link = tmp.resolve("root-link");
    Files.createSymbolicLink(link, real);
    LocalFileStorageBackend backend = new LocalFileStorageBackend(link);

    backend.put("a/b.bin", new byte[] {7});
    assertThat(Files.exists(real.resolve("a/b.bin"))).isTrue();
    assertThat(backend.get("a/b.bin")).containsExactly(7);
  }

  @Test
  void nestedKeysStayUnderTheRootAndRoundTrip() throws IOException {
    Path root = root();
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root);
    backend.put("tenant/coll/gen-0001/vectors.bin", new byte[] {1, 2, 3});
    assertThat(Files.exists(root.resolve("tenant/coll/gen-0001/vectors.bin"))).isTrue();
    assertThat(backend.get("tenant/coll/gen-0001/vectors.bin")).containsExactly(1, 2, 3);
    assertThat(backend.list("tenant/coll/")).containsExactly("tenant/coll/gen-0001/vectors.bin");
  }

  @Test
  void keysMayNotNameTheBackendsOwnTemporaryOrEtagFiles() throws IOException {
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root());
    assertThatThrownBy(() -> backend.put("a.tmp", new byte[] {1})).isInstanceOf(IOException.class);
    assertThatThrownBy(() -> backend.put("a.etag", new byte[] {1})).isInstanceOf(IOException.class);
  }

  /**
   * A listing that runs while another thread writes and removes objects must never throw: the
   * writer's temporary files appear and vanish between the directory read and the attribute read,
   * and a walk that trips on one takes the whole listing down (seen on DartVault's HC4 harness on
   * 2026-11-17: a backup listed while a replica put was in flight).
   */
  @Test
  void listingWhileAnotherThreadWritesAndRemovesNeverThrows() throws Exception {
    java.nio.file.Path root = java.nio.file.Files.createTempDirectory("list-race-");
    LocalFileStorageBackend backend = new LocalFileStorageBackend(root);
    java.util.concurrent.atomic.AtomicBoolean on =
        new java.util.concurrent.atomic.AtomicBoolean(true);
    java.util.concurrent.atomic.AtomicReference<Throwable> writerFailure =
        new java.util.concurrent.atomic.AtomicReference<>();
    Thread writer =
        new Thread(
            () -> {
              byte[] v = new byte[64];
              int i = 0;
              try {
                while (on.get()) {
                  String key = "race/coll/obj" + (i++ % 8);
                  backend.put(key, v);
                  backend.delete(key);
                }
              } catch (Throwable t) {
                writerFailure.set(t);
              }
            },
            "list-race-writer");
    writer.start();
    long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(3);
    int listings = 0;
    try {
      while (System.nanoTime() < deadline) {
        backend.list("race/");
        listings++;
      }
    } finally {
      on.set(false);
      writer.join(5_000);
    }
    org.assertj.core.api.Assertions.assertThat(writerFailure.get()).isNull();
    org.assertj.core.api.Assertions.assertThat(listings).isPositive();
  }
}
