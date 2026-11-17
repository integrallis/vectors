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
package com.integrallis.vectors.storage.testkit;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * Plain writes and reads: round trip, defensive copies, missing keys, overwrite, streaming. Part of
 * {@link StorageBackendContract}.
 */
public interface ReadWriteChecks extends BackendUnderTest {

  @Test
  default void putAndGetRoundTrip() throws IOException {
    StorageBackend b = backend();
    byte[] value = "hello world".getBytes(StandardCharsets.UTF_8);
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
  default void overwriteWithPutChangesValue() throws IOException {
    StorageBackend b = backend();
    String key = key("overwrite");
    b.put(key, new byte[] {1, 2, 3});
    b.put(key, new byte[] {9});
    assertThat(b.get(key)).isEqualTo(new byte[] {9});
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
}
