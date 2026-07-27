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
package com.integrallis.vectors.storage.manifest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class StorageManifestTest {

  private static StorageManifest sample() {
    Map<String, Long> e = new LinkedHashMap<>();
    e.put("shard-0", 7L);
    e.put("shard-1", 12L);
    return new StorageManifest(42L, "sha256-abc", e, 1_700_000_000_000L, "writer-a");
  }

  @Test
  void encodeDecodeRoundTrips() {
    StorageManifest m = sample();
    StorageManifest back = StorageManifest.decode(m.encode());
    assertThat(back).isEqualTo(m);
    assertThat(back.generation()).isEqualTo(42L);
    assertThat(back.contentHash()).isEqualTo("sha256-abc");
    assertThat(back.entries()).containsExactly(Map.entry("shard-0", 7L), Map.entry("shard-1", 12L));
    assertThat(back.committedAtEpochMs()).isEqualTo(1_700_000_000_000L);
    assertThat(back.writer()).isEqualTo("writer-a");
  }

  @Test
  void emptyManifestRoundTrips() {
    StorageManifest back = StorageManifest.decode(StorageManifest.EMPTY.encode());
    assertThat(back).isEqualTo(StorageManifest.EMPTY);
    assertThat(back.isEmpty()).isTrue();
    assertThat(back.entries()).isEmpty();
  }

  @Test
  void decodeRejectsBadMagicAndTruncation() {
    assertThatThrownBy(() -> StorageManifest.decode(new byte[] {1, 2, 3, 4, 5}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("magic");
    byte[] good = sample().encode();
    byte[] truncated = new byte[good.length - 4];
    System.arraycopy(good, 0, truncated, 0, truncated.length);
    assertThatThrownBy(() -> StorageManifest.decode(truncated))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void atGenerationAndWithEntryAreImmutableCopies() {
    StorageManifest m = sample();
    StorageManifest g = m.atGeneration(100L);
    assertThat(g.generation()).isEqualTo(100L);
    assertThat(m.generation()).isEqualTo(42L); // original untouched
    assertThat(g.entries()).isEqualTo(m.entries());

    StorageManifest e = m.withEntry("shard-2", 3L);
    assertThat(e.entries()).containsEntry("shard-2", 3L).containsEntry("shard-0", 7L);
    assertThat(m.entries()).doesNotContainKey("shard-2"); // original untouched
  }

  @Test
  void entriesAreUnmodifiable() {
    StorageManifest m = sample();
    assertThatThrownBy(() -> m.entries().put("x", 1L))
        .isInstanceOf(UnsupportedOperationException.class);
  }
}
