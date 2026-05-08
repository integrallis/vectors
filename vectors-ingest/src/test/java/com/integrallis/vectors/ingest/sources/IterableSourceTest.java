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
package com.integrallis.vectors.ingest.sources;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.ingest.IngestDoc;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IterableSourceTest {

  @Test
  void iteratesAllDocs() {
    List<IngestDoc> docs =
        List.of(IngestDoc.text("a", "1"), IngestDoc.text("b", "2"), IngestDoc.text("c", "3"));
    IterableSource src = IterableSource.of("test", docs);
    List<String> ids = new ArrayList<>();
    src.forEach(d -> ids.add(d.id()));
    assertThat(ids).containsExactly("a", "b", "c");
    assertThat(src.name()).isEqualTo("test");
    assertThat(src.estimatedSize()).hasValue(3);
    assertThat(src.startOffset()).isEqualTo(0L);
  }

  @Test
  void resumesFromOffset() {
    List<IngestDoc> docs =
        List.of(IngestDoc.text("a", "1"), IngestDoc.text("b", "2"), IngestDoc.text("c", "3"));
    IterableSource src = IterableSource.resuming("test", docs, 2);
    List<String> ids = new ArrayList<>();
    src.forEach(d -> ids.add(d.id()));
    assertThat(ids).containsExactly("c");
    assertThat(src.startOffset()).isEqualTo(2L);
  }

  @Test
  void offsetBeyondSizeYieldsEmpty() {
    IterableSource src = IterableSource.resuming("test", List.of(IngestDoc.text("a", "1")), 10);
    List<String> ids = new ArrayList<>();
    src.forEach(d -> ids.add(d.id()));
    assertThat(ids).isEmpty();
  }

  @Test
  void emptySourceWorks() {
    IterableSource src = IterableSource.copyOf("e", List.of());
    assertThat(src.iterator().hasNext()).isFalse();
    assertThat(src.estimatedSize()).hasValue(0);
  }
}
