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
package com.integrallis.vectors.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BatchTest {

  @Test
  void preservesOrderAndOffsets() {
    List<EmbeddedDoc> docs =
        List.of(
            new EmbeddedDoc(IngestDoc.text("a", "1"), new float[] {0f}, 0L),
            new EmbeddedDoc(IngestDoc.text("b", "2"), new float[] {0f}, 1L),
            new EmbeddedDoc(IngestDoc.text("c", "3"), new float[] {0f}, 2L));
    Batch b = new Batch(7, docs);
    assertThat(b.batchId()).isEqualTo(7);
    assertThat(b.size()).isEqualTo(3);
    assertThat(b.lastSourceOffset()).isEqualTo(2L);
    assertThat(b.docs().get(0).doc().id()).isEqualTo("a");
  }

  @Test
  void docsListIsImmutable() {
    List<EmbeddedDoc> mutable = new ArrayList<>();
    mutable.add(new EmbeddedDoc(IngestDoc.text("a", "1"), new float[] {0f}, 0L));
    Batch b = new Batch(0, mutable);
    assertThatThrownBy(() -> b.docs().add(null)).isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void rejectsEmpty() {
    assertThatThrownBy(() -> new Batch(0, List.of())).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNegativeBatchId() {
    assertThatThrownBy(
            () ->
                new Batch(
                    -1, List.of(new EmbeddedDoc(IngestDoc.text("a", "1"), new float[] {0f}, 0L))))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
