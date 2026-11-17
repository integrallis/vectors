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
import org.junit.jupiter.api.Test;

/**
 * Etags and compare-and-swap: a read returns the current etag, and a conditional put honours it.
 * Part of {@link StorageBackendContract}.
 */
public interface EtagChecks extends BackendUnderTest {

  @Test
  default void getWithEtagReturnsValueAndCurrentEtag() throws IOException {
    StorageBackend b = backend();
    String key = key("get-etag");
    StorageBackend.ConditionalPutResult written = b.conditionalPut(key, new byte[] {1, 2}, null);

    StorageBackend.StoredValue stored = b.getWithEtag(key);

    assertThat(stored).isNotNull();
    assertThat(stored.value()).isEqualTo(new byte[] {1, 2});
    assertThat(stored.etag()).isEqualTo(written.newEtag());
  }

  @Test
  default void getWithEtagReturnsDefensiveCopy() throws IOException {
    StorageBackend b = backend();
    String key = key("copy-get-etag");
    b.put(key, new byte[] {1, 2, 3});

    StorageBackend.StoredValue first = b.getWithEtag(key);
    first.value()[0] = 99;

    assertThat(b.getWithEtag(key).value()).isEqualTo(new byte[] {1, 2, 3});
  }

  @Test
  default void getWithEtagMissingKeyReturnsNull() throws IOException {
    assertThat(backend().getWithEtag(key("nonexistent-etag"))).isNull();
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
}
