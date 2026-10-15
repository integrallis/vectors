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
package com.integrallis.vectors.vcr;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class LocalVcrStorageBackendTest {
  @TempDir Path root;

  @Test
  void portableNamesPreserveLogicalKeysWithoutCollisions() throws Exception {
    var backend = new LocalVcrStorageBackend(root);
    List<String> keys =
        List.of(
            "vcr:embedding:Case:0001",
            "vcr:embedding:case:0001",
            "vcr:registry:test:CON",
            "vcr:registry:test:a/b\\c:<d>*?\"|. ",
            "vcr:registry:test:é😀",
            "vcr:registry:test:%003a",
            "vcr:registry:test:" + "x".repeat(150));
    for (int i = 0; i < keys.size(); i++) backend.put(keys.get(i), new byte[] {(byte) i, 7});
    var reopened = new LocalVcrStorageBackend(root);
    assertEquals(new java.util.HashSet<>(keys), new java.util.HashSet<>(reopened.list("vcr:")));
    for (int i = 0; i < keys.size(); i++) {
      assertArrayEquals(new byte[] {(byte) i, 7}, reopened.get(keys.get(i)));
      assertArrayEquals(new byte[] {7}, reopened.getRange(keys.get(i), 1, 1));
    }
    try (var paths = Files.walk(root)) {
      paths
          .filter(Files::isRegularFile)
          .forEach(
              path -> {
                for (Path part : root.relativize(path)) {
                  assertTrue(part.toString().matches("[a-z0-9-]+"), part.toString());
                  assertTrue(part.toString().length() <= 128);
                }
              });
    }
    assertEquals(2, reopened.list("vcr:embedding:").size());
    for (String key : keys) reopened.delete(key);
    assertTrue(reopened.list("vcr:").isEmpty());
    reopened.delete("vcr:registry:absent");
  }

  @Test
  void registryAndConditionalWritesSurviveReopen() throws Exception {
    var backend = new LocalVcrStorageBackend(root);
    var registry = new VCRRegistry(backend);
    registry.registerSuccess("Class:method");
    assertEquals(
        VCRRegistry.RecordingStatus.RECORDED,
        new VCRRegistry(new LocalVcrStorageBackend(root)).getTestStatus("Class:method"));
    String key = "vcr:embedding:Class:method:0001";
    var first = backend.conditionalPut(key, new byte[] {1}, null);
    assertTrue(first.succeeded());
    assertFalse(backend.conditionalPut(key, new byte[] {2}, null).succeeded());
    assertTrue(backend.conditionalPut(key, new byte[] {3}, first.newEtag()).succeeded());
    assertArrayEquals(new byte[] {3}, backend.get(key));
  }

  @Test
  void legacyUnixRecordingsMigrateWithoutChangingIdentity() throws Exception {
    String key = "vcr:embedding:Legacy:method:0001";
    try {
      Path.of(key);
    } catch (InvalidPathException windowsCannotRepresentLegacy) {
      assertNull(new LocalVcrStorageBackend(root).get(key));
      return;
    }
    var old = new LocalFileStorageBackend(root);
    old.put(key, new byte[] {1, 2});
    var backend = new LocalVcrStorageBackend(root);
    assertArrayEquals(new byte[] {1, 2}, backend.get(key));
    String etag = backend.getWithEtag(key).etag();
    assertTrue(backend.conditionalPut(key, new byte[] {3}, etag).succeeded());
    assertFalse(Files.exists(root.resolve(key)));
    assertArrayEquals(new byte[] {3}, backend.get(key));
    old.put(key, new byte[] {9});
    assertArrayEquals(new byte[] {3}, backend.get(key));
    assertEquals(List.of(key), backend.list("vcr:embedding:"));
    backend.delete(key);
    assertNull(old.get(key));
    assertNull(backend.get(key));
    String nested = "vcr:embedding:Legacy/nested:0001";
    old.put(nested, new byte[] {8});
    assertArrayEquals(new byte[] {8}, backend.get(nested));
    assertEquals(List.of(nested), backend.list("vcr:embedding:"));
    backend.delete(nested);
    assertNull(old.get(nested));
    Files.write(root.resolve(key), new byte[] {4, 5});
    assertArrayEquals(new byte[] {4, 5}, backend.get(key));
    backend.delete(key);
    assertFalse(Files.exists(root.resolve(key)));
  }
}
