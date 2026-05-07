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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Contract test for all in-JVM {@link StorageBackend} implementations. Each parameterised test runs
 * against {@link HeapStorageBackend} and {@link LocalFileStorageBackend}.
 */
@Tag("unit")
class StorageBackendContractTest {

  @TempDir Path tmpDir;

  static Stream<String> implementations() {
    return Stream.of("heap", "local");
  }

  private StorageBackend backend(String type) throws IOException {
    return switch (type) {
      case "heap" -> new HeapStorageBackend();
      case "local" -> new LocalFileStorageBackend(Files.createTempDirectory(null));
      default -> throw new IllegalArgumentException(type);
    };
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void putAndGetRoundTrip(String type) throws IOException {
    StorageBackend b = backend(type);
    byte[] value = "hello world".getBytes();
    b.put("mykey", value);
    assertThat(b.get("mykey")).isEqualTo(value);
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getMissingKeyReturnsNull(String type) throws IOException {
    assertThat(backend(type).get("nonexistent")).isNull();
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void listReturnsKeysUnderPrefix(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("a/x", new byte[] {1});
    b.put("a/y", new byte[] {2});
    b.put("b/z", new byte[] {3});

    List<String> result = b.list("a/");
    assertThat(result).hasSize(2).containsExactlyInAnyOrder("a/x", "a/y");
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void deleteRemovesKey(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("todelete", new byte[] {42});
    b.delete("todelete");
    assertThat(b.get("todelete")).isNull();
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void deleteMissingKeyIsNoOp(String type) throws IOException {
    backend(type).delete("ghost"); // must not throw
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void conditionalPut_succeedsWhenKeyAbsent(String type) throws IOException {
    StorageBackend b = backend(type);
    StorageBackend.ConditionalPutResult r =
        b.conditionalPut("newkey", new byte[] {1, 2}, null /* must not exist */);

    assertThat(r.succeeded()).isTrue();
    assertThat(r.newEtag()).isNotNull().isNotEmpty();
    assertThat(b.get("newkey")).isEqualTo(new byte[] {1, 2});
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void conditionalPut_failsWhenKeyExistsAndEtagIsNull(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("exists", new byte[] {1});
    StorageBackend.ConditionalPutResult r = b.conditionalPut("exists", new byte[] {2}, null);

    assertThat(r.succeeded()).isFalse();
    assertThat(b.get("exists")).isEqualTo(new byte[] {1}); // unchanged
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void conditionalPut_failsOnStaleEtag(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("k", new byte[] {10});
    StorageBackend.ConditionalPutResult r =
        b.conditionalPut("k", new byte[] {20}, "stale-etag-xxx");

    assertThat(r.succeeded()).isFalse();
    assertThat(b.get("k")).isEqualTo(new byte[] {10});
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void conditionalPut_succeedsWithCurrentEtag(String type) throws IOException {
    StorageBackend b = backend(type);
    StorageBackend.ConditionalPutResult r1 = b.conditionalPut("chain", new byte[] {1}, null);
    assertThat(r1.succeeded()).isTrue();

    StorageBackend.ConditionalPutResult r2 =
        b.conditionalPut("chain", new byte[] {2}, r1.newEtag());
    assertThat(r2.succeeded()).isTrue();
    assertThat(b.get("chain")).isEqualTo(new byte[] {2});

    StorageBackend.ConditionalPutResult r3 =
        b.conditionalPut("chain", new byte[] {3}, r2.newEtag());
    assertThat(r3.succeeded()).isTrue();
    assertThat(b.get("chain")).isEqualTo(new byte[] {3});
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void overwriteWithPutChangesValue(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("key", new byte[] {1, 2, 3});
    b.put("key", new byte[] {9});
    assertThat(b.get("key")).isEqualTo(new byte[] {9});
  }

  // ─── getRange (§6.1) ──────────────────────────────────────────────────────

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_partialFetchMatchesFullSlice(String type) throws IOException {
    StorageBackend b = backend(type);
    byte[] full = new byte[1024];
    for (int i = 0; i < full.length; i++) full[i] = (byte) (i & 0xFF);
    b.put("blob", full);

    byte[] slice = b.getRange("blob", 100, 256);

    assertThat(slice).hasSize(256).isEqualTo(Arrays.copyOfRange(full, 100, 356));
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_offsetZeroLengthEqualsSize_returnsFullValue(String type) throws IOException {
    StorageBackend b = backend(type);
    byte[] full = "the quick brown fox".getBytes();
    b.put("k", full);

    assertThat(b.getRange("k", 0, full.length)).isEqualTo(full);
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_zeroLengthReturnsEmpty(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("k", new byte[] {1, 2, 3});

    assertThat(b.getRange("k", 1, 0)).isEmpty();
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_missingKeyReturnsNull(String type) throws IOException {
    assertThat(backend(type).getRange("ghost", 0, 4)).isNull();
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_pastEofThrows(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("k", new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange("k", 2, 5))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_negativeOffsetThrows(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("k", new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange("k", -1, 2))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @ParameterizedTest
  @MethodSource("implementations")
  void getRange_negativeLengthThrows(String type) throws IOException {
    StorageBackend b = backend(type);
    b.put("k", new byte[] {1, 2, 3, 4});

    assertThatThrownBy(() -> b.getRange("k", 0, -1))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }
}
