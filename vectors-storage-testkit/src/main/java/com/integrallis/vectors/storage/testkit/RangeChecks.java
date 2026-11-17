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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Ranged reads: partial fetches, bounds, overflow and the error cases the interface specifies. Part
 * of {@link StorageBackendContract}.
 */
public interface RangeChecks extends BackendUnderTest {

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
    byte[] full = "the quick brown fox".getBytes(StandardCharsets.UTF_8);
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
  default void getRange_hugeOffsetCannotOverflow() throws IOException {
    StorageBackend b = backend();
    String key = key("range-overflow");
    b.put(key, new byte[] {1, 2, 3});
    assertThatThrownBy(() -> b.getRange(key, Long.MAX_VALUE, 2))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  default void getRange_zeroLengthStillChecksOffset() throws IOException {
    StorageBackend b = backend();
    String key = key("range-empty-bounds");
    b.put(key, new byte[] {1, 2, 3});
    assertThat(b.getRange(key, 3, 0)).isEmpty();
    assertThatThrownBy(() -> b.getRange(key, 4, 0)).isInstanceOf(IndexOutOfBoundsException.class);
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
