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
package com.integrallis.vectors.ingest.cursor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.util.ConcurrentModificationException;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class R2KeyCursorTest {

  @Test
  void unknownSourceReturnsZeroOnFreshBackend() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor c = new R2KeyCursor(backend);
    assertThat(c.load("any")).isEqualTo(0L);
    assertThat(backend.get(R2KeyCursor.DEFAULT_KEY)).isNull();
  }

  @Test
  void roundTripsAcrossInstances() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor a = new R2KeyCursor(backend);
    a.save("src", 42L);
    a.save("other", 7L);
    R2KeyCursor b = new R2KeyCursor(backend);
    assertThat(b.load("src")).isEqualTo(42L);
    assertThat(b.load("other")).isEqualTo(7L);
    assertThat(b.load("missing")).isEqualTo(0L);
  }

  @Test
  void firstSaveUsesConditionalPutNullThenRefreshesEtag() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor c = new R2KeyCursor(backend);
    c.save("s", 1L);
    String e1 = c.currentEtag();
    assertThat(e1).isNotNull();
    c.save("s", 2L);
    String e2 = c.currentEtag();
    assertThat(e2).isNotNull().isNotEqualTo(e1);
  }

  @Test
  void casConflictThrowsConcurrentModification() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor a = new R2KeyCursor(backend);
    a.save("s", 10L); // a now caches a valid etag
    // A second writer arrives, observes the existing key, and writes via the
    // unconditional fallback path → bumps the backend etag.
    R2KeyCursor b = new R2KeyCursor(backend);
    b.save("s", 99L);
    // a's cached etag is now stale → next CAS save must blow up.
    assertThatThrownBy(() -> a.save("s", 11L))
        .isInstanceOf(ConcurrentModificationException.class)
        .hasMessageContaining(R2KeyCursor.DEFAULT_KEY);
  }

  @Test
  void concurrentFreshCursorsRaceFirstSave() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor a = new R2KeyCursor(backend);
    R2KeyCursor b = new R2KeyCursor(backend);
    a.save("s", 5L);
    // b's first-save CAS-null fails (key now exists); falls back to unconditional put → no throw.
    b.save("s", 99L);
    assertThat(new R2KeyCursor(backend).load("s")).isEqualTo(99L);
  }

  @Test
  void honorsCustomKey() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor c = new R2KeyCursor(backend, "custom/cursor.json");
    c.save("s", 12L);
    assertThat(backend.get("custom/cursor.json")).isNotNull();
    assertThat(backend.get(R2KeyCursor.DEFAULT_KEY)).isNull();
    assertThat(c.key()).isEqualTo("custom/cursor.json");
  }

  @Test
  void snapshotJsonReflectsState() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor c = new R2KeyCursor(backend);
    c.save("a", 1L);
    c.save("b", 2L);
    String json = c.snapshotJson();
    assertThat(json).contains("\"a\":1").contains("\"b\":2");
  }

  @Test
  void rejectsBadConstructorArgs() {
    HeapStorageBackend backend = new HeapStorageBackend();
    assertThatThrownBy(() -> new R2KeyCursor(backend, " "))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new R2KeyCursor(null)).isInstanceOf(NullPointerException.class);
  }

  @Test
  void rejectsNegativeOffset() {
    HeapStorageBackend backend = new HeapStorageBackend();
    R2KeyCursor c = new R2KeyCursor(backend);
    assertThatThrownBy(() -> c.save("s", -1L)).isInstanceOf(IllegalArgumentException.class);
  }
}
