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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EmbeddedDocTest {

  @Test
  void buildsHappyPath() {
    EmbeddedDoc e = new EmbeddedDoc(IngestDoc.text("a", "x"), new float[] {1f, 2f}, 5L);
    assertThat(e.doc().id()).isEqualTo("a");
    assertThat(e.vector()).containsExactly(1f, 2f);
    assertThat(e.sourceOffset()).isEqualTo(5L);
  }

  @Test
  void rejectsEmptyVector() {
    assertThatThrownBy(() -> new EmbeddedDoc(IngestDoc.text("a", "x"), new float[0], 0L))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNegativeOffset() {
    assertThatThrownBy(() -> new EmbeddedDoc(IngestDoc.text("a", "x"), new float[] {1f}, -1L))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
