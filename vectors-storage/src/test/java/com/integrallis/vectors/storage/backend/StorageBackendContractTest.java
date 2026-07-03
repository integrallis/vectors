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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class HeapStorageBackendContractTest implements StorageBackendContract {
  @Override
  public StorageBackend backend() {
    return new HeapStorageBackend();
  }

  @Test
  void heapEtagIsSha256HexOfStoredValue() throws IOException {
    StorageBackend.ConditionalPutResult result =
        backend().conditionalPut(key("sha256-etag"), new byte[] {1, 2, 3}, null);

    assertThat(result.succeeded()).isTrue();
    assertThat(result.newEtag())
        .isEqualTo("039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81");
  }
}

@Tag("unit")
class LocalFileStorageBackendContractTest implements StorageBackendContract {
  @TempDir Path tmpDir;

  @Override
  public StorageBackend backend() throws IOException {
    return new LocalFileStorageBackend(tmpDir);
  }

  @Test
  void localFileRejectsPathTraversal() throws IOException {
    StorageBackend b = backend();

    assertThatThrownBy(() -> b.put("../escape", new byte[] {1})).isInstanceOf(IOException.class);
    assertThatThrownBy(() -> b.put("a/../escape", new byte[] {1})).isInstanceOf(IOException.class);
    assertThatThrownBy(() -> b.get("/absolute")).isInstanceOf(IOException.class);
    assertThat(Files.exists(tmpDir.resolveSibling("escape"))).isFalse();
  }

  @Test
  void localFileStoresEtagAndValueAtomicallyInOneFile() throws IOException {
    StorageBackend b = backend();
    String key = key("atomic");

    StorageBackend.ConditionalPutResult first = b.conditionalPut(key, new byte[] {1, 2, 3}, null);
    StorageBackend.ConditionalPutResult second =
        b.conditionalPut(key, new byte[] {4, 5, 6}, first.newEtag());

    assertThat(first.succeeded()).isTrue();
    assertThat(second.succeeded()).isTrue();
    assertThat(b.get(key)).isEqualTo(new byte[] {4, 5, 6});

    StorageBackend reopened = new LocalFileStorageBackend(tmpDir);
    assertThat(reopened.get(key)).isEqualTo(new byte[] {4, 5, 6});
    StorageBackend.ConditionalPutResult stale =
        reopened.conditionalPut(key, new byte[] {7}, first.newEtag());
    assertThat(stale.succeeded()).isFalse();
    assertThat(reopened.get(key)).isEqualTo(new byte[] {4, 5, 6});
  }
}
