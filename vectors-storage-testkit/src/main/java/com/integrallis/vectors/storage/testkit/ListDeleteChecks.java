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
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Listing by prefix and deletion, including the no-op delete of a missing key. Part of {@link
 * StorageBackendContract}.
 */
public interface ListDeleteChecks extends BackendUnderTest {

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
}
