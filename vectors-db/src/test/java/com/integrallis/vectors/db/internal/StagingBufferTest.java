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
package com.integrallis.vectors.db.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.db.Document;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class StagingBufferTest {

  private static Document doc(String id) {
    return Document.of(id, new float[] {1f, 2f, 3f, 4f});
  }

  @Test
  void emptyBufferReportsZero() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.size()).isZero();
    assertThat(buf.isEmpty()).isTrue();
    assertThat(buf.contains("a")).isFalse();
    assertThat(buf.documents()).isEmpty();
  }

  @Test
  void appendTracksSizeAndOrder() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.append(doc("a"))).isTrue();
    assertThat(buf.append(doc("b"))).isTrue();
    assertThat(buf.size()).isEqualTo(2);
    assertThat(buf.isEmpty()).isFalse();
    assertThat(buf.contains("a")).isTrue();
    assertThat(buf.contains("b")).isTrue();
    assertThat(buf.documents().get(0).id()).isEqualTo("a");
    assertThat(buf.documents().get(1).id()).isEqualTo("b");
  }

  @Test
  void appendDuplicateReturnsFalseAndDoesNotMutate() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    assertThat(buf.append(doc("a"))).isFalse();
    assertThat(buf.size()).isEqualTo(1);
    assertThat(buf.documents()).hasSize(1);
  }

  @Test
  void clearWipesDocumentsAndIds() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    buf.append(doc("b"));
    buf.clear();
    assertThat(buf.size()).isZero();
    assertThat(buf.isEmpty()).isTrue();
    assertThat(buf.contains("a")).isFalse();
    // After clear the same ids can be re-appended.
    assertThat(buf.append(doc("a"))).isTrue();
  }

  @Test
  void appendNullThrows() {
    StagingBuffer buf = new StagingBuffer();
    assertThatNullPointerException().isThrownBy(() -> buf.append(null));
  }

  // ---------------------------------------------------------------------------
  // Tombstone support (Step 6)
  // ---------------------------------------------------------------------------

  @Test
  void hasWorkFalseWhenEmpty() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.hasWork()).isFalse();
  }

  @Test
  void hasWorkTrueWithStagedDocuments() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    assertThat(buf.hasWork()).isTrue();
  }

  @Test
  void hasWorkTrueWithPendingTombstones() {
    StagingBuffer buf = new StagingBuffer();
    buf.stageDelete("x");
    assertThat(buf.hasWork()).isTrue();
  }

  @Test
  void stageDeleteTracksIds() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.stageDelete("x")).isTrue();
    assertThat(buf.isTombstoned("x")).isTrue();
    assertThat(buf.isTombstoned("y")).isFalse();
    assertThat(buf.pendingTombstones()).containsExactly("x");
  }

  @Test
  void stageDeleteDuplicateReturnsFalse() {
    StagingBuffer buf = new StagingBuffer();
    buf.stageDelete("x");
    assertThat(buf.stageDelete("x")).isFalse();
    assertThat(buf.pendingTombstones()).hasSize(1);
  }

  @Test
  void stageDeleteNullThrows() {
    StagingBuffer buf = new StagingBuffer();
    assertThatNullPointerException().isThrownBy(() -> buf.stageDelete(null));
  }

  @Test
  void removeDocumentReturnsTrueAndRemoves() {
    StagingBuffer buf = new StagingBuffer();
    buf.append(doc("a"));
    buf.append(doc("b"));
    assertThat(buf.removeDocument("a")).isTrue();
    assertThat(buf.size()).isEqualTo(1);
    assertThat(buf.contains("a")).isFalse();
    assertThat(buf.documents()).hasSize(1);
    assertThat(buf.documents().get(0).id()).isEqualTo("b");
  }

  @Test
  void removeDocumentReturnsFalseForUnknownId() {
    StagingBuffer buf = new StagingBuffer();
    assertThat(buf.removeDocument("x")).isFalse();
  }

  @Test
  void removeDocumentNullThrows() {
    StagingBuffer buf = new StagingBuffer();
    assertThatNullPointerException().isThrownBy(() -> buf.removeDocument(null));
  }

  @Test
  void clearWipesPendingTombstones() {
    StagingBuffer buf = new StagingBuffer();
    buf.stageDelete("x");
    buf.stageDelete("y");
    buf.clear();
    assertThat(buf.isTombstoned("x")).isFalse();
    assertThat(buf.pendingTombstones()).isEmpty();
    assertThat(buf.hasWork()).isFalse();
  }

  @Test
  void pendingTombstonesIsUnmodifiable() {
    StagingBuffer buf = new StagingBuffer();
    buf.stageDelete("x");
    var tombstones = buf.pendingTombstones();
    org.assertj.core.api.Assertions.assertThatExceptionOfType(UnsupportedOperationException.class)
        .isThrownBy(() -> tombstones.add("y"));
  }
}
