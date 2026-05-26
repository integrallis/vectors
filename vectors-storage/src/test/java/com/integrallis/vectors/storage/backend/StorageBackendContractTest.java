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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

interface StorageBackendContract {

  StorageBackend backend() throws IOException;

  default String key(String suffix) {
    return suffix + "-" + UUID.randomUUID();
  }

  @Test
  default void putAndGetRoundTrip() throws IOException {
    StorageBackend b = backend();
    byte[] value = "hello world".getBytes();
    String key = key("mykey");
    b.put(key, value);
    assertThat(b.get(key)).isEqualTo(value);
  }

  @Test
  default void putDefensivelyCopiesInputArray() throws IOException {
    StorageBackend b = backend();
    String key = key("copy-put");
    byte[] value = new byte[] {1, 2, 3};

    b.put(key, value);
    value[0] = 99;

    assertThat(b.get(key)).isEqualTo(new byte[] {1, 2, 3});
  }

  @Test
  default void getReturnsDefensiveCopy() throws IOException {
    StorageBackend b = backend();
    String key = key("copy-get");
    b.put(key, new byte[] {1, 2, 3});

    byte[] first = b.get(key);
    first[0] = 99;

    assertThat(b.get(key)).isEqualTo(new byte[] {1, 2, 3});
  }

  @Test
  default void getMissingKeyReturnsNull() throws IOException {
    assertThat(backend().get(key("nonexistent"))).isNull();
  }

  @Test
  default void openStreamsStoredValue() throws IOException {
    StorageBackend b = backend();
    String key = key("open");
    b.put(key, new byte[] {1, 2, 3, 4});

    try (InputStream in = b.open(key)) {
      assertThat(in).isNotNull();
      assertThat(in.readAllBytes()).isEqualTo(new byte[] {1, 2, 3, 4});
    }
  }

  @Test
  default void openMissingKeyReturnsNull() throws IOException {
    assertThat(backend().open(key("open-missing"))).isNull();
  }

  @Test
  default void listReturnsKeysUnderPrefix() throws IOException {
    StorageBackend b = backend();
    String prefix = key("list") + "/";
    b.put(prefix + "a", new byte[] {1});
    b.put(prefix + "b", new byte[] {2});
    b.put(key("other"), new byte[] {3});

    List<String> result = b.list(prefix);

    assertThat(result).hasSize(2).containsExactlyInAnyOrder(prefix + "a", prefix + "b");
  }

  @Test
  default void deleteRemovesKey() throws IOException {
    StorageBackend b = backend();
    String key = key("delete");
    b.put(key, new byte[] {42});
    b.delete(key);
    assertThat(b.get(key)).isNull();
  }

  @Test
  default void deleteMissingKeyIsNoOp() throws IOException {
    backend().delete(key("ghost"));
  }

  @Test
  default void conditionalPut_succeedsWhenKeyAbsent() throws IOException {
    StorageBackend b = backend();
    String key = key("cas-new");
    StorageBackend.ConditionalPutResult r = b.conditionalPut(key, new byte[] {1, 2}, null);

    assertThat(r.succeeded()).isTrue();
    assertThat(r.newEtag()).isNotNull().isNotEmpty();
    assertThat(b.get(key)).isEqualTo(new byte[] {1, 2});
  }

  @Test
  default void conditionalPut_failsWhenKeyExistsAndEtagIsNull() throws IOException {
    StorageBackend b = backend();
    String key = key("cas-exists");
    b.put(key, new byte[] {1});
    StorageBackend.ConditionalPutResult r = b.conditionalPut(key, new byte[] {2}, null);

    assertThat(r.succeeded()).isFalse();
    assertThat(b.get(key)).isEqualTo(new byte[] {1});
  }

  @Test
  default void conditionalPut_failsOnStaleEtag() throws IOException {
    StorageBackend b = backend();
    String key = key("cas-stale");
    b.put(key, new byte[] {10});
    StorageBackend.ConditionalPutResult r =
        b.conditionalPut(key, new byte[] {20}, "stale-etag-xxx");

    assertThat(r.succeeded()).isFalse();
    assertThat(b.get(key)).isEqualTo(new byte[] {10});
  }

  @Test
  default void conditionalPut_succeedsWithCurrentEtag() throws IOException {
    StorageBackend b = backend();
    String key = key("cas-current");
    StorageBackend.ConditionalPutResult r1 = b.conditionalPut(key, new byte[] {1}, null);
    assertThat(r1.succeeded()).isTrue();

    StorageBackend.ConditionalPutResult r2 = b.conditionalPut(key, new byte[] {2}, r1.newEtag());
    assertThat(r2.succeeded()).isTrue();
    assertThat(b.get(key)).isEqualTo(new byte[] {2});

    StorageBackend.ConditionalPutResult r3 = b.conditionalPut(key, new byte[] {3}, r2.newEtag());
    assertThat(r3.succeeded()).isTrue();
    assertThat(b.get(key)).isEqualTo(new byte[] {3});
  }

  @Test
  default void overwriteWithPutChangesValue() throws IOException {
    StorageBackend b = backend();
    String key = key("overwrite");
    b.put(key, new byte[] {1, 2, 3});
    b.put(key, new byte[] {9});
    assertThat(b.get(key)).isEqualTo(new byte[] {9});
  }

  @Test
  default void getRange_partialFetchMatchesFullSlice() throws IOException {
    StorageBackend b = backend();
    String key = key("range");
    byte[] full = new byte[1024];
    for (int i = 0; i < full.length; i++) full[i] = (byte) (i & 0xFF);
    b.put(key, full);

    byte[] slice = b.getRange(key, 100, 256);

    assertThat(slice).hasSize(256).isEqualTo(Arrays.copyOfRange(full, 100, 356));
  }

  @Test
  default void getRange_offsetZeroLengthEqualsSize_returnsFullValue() throws IOException {
    StorageBackend b = backend();
    String key = key("range-full");
    byte[] full = "the quick brown fox".getBytes();
    b.put(key, full);

    assertThat(b.getRange(key, 0, full.length)).isEqualTo(full);
  }

  @Test
  default void getRange_zeroLengthReturnsEmpty() throws IOException {
    StorageBackend b = backend();
    String key = key("range-empty");
    b.put(key, new byte[] {1, 2, 3});

    assertThat(b.getRange(key, 1, 0)).isEmpty();
  }

  @Test
  default void getRange_missingKeyReturnsNull() throws IOException {
    assertThat(backend().getRange(key("range-missing"), 0, 4)).isNull();
  }

  @Test
  default void getRange_pastEofThrows() throws IOException {
    StorageBackend b = backend();
    String key = key("range-past");
    b.put(key, new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange(key, 2, 5)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  default void getRange_negativeOffsetThrows() throws IOException {
    StorageBackend b = backend();
    String key = key("range-negative-offset");
    b.put(key, new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange(key, -1, 2)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  default void getRange_negativeLengthThrows() throws IOException {
    StorageBackend b = backend();
    String key = key("range-negative-length");
    b.put(key, new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange(key, 0, -1)).isInstanceOf(IndexOutOfBoundsException.class);
  }
}

@Tag("unit")
class HeapStorageBackendContractTest implements StorageBackendContract {
  @Override
  public StorageBackend backend() {
    return new HeapStorageBackend();
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
