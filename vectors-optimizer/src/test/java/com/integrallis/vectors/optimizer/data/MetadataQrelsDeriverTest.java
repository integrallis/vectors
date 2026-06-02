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
package com.integrallis.vectors.optimizer.data;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MetadataQrelsDeriverTest {

  private static Document doc(String id, String labelField, MetadataValue v) {
    return new Document(id, new float[] {0f, 0f}, null, Map.of(labelField, v));
  }

  private static Document docNoMeta(String id) {
    return new Document(id, new float[] {0f, 0f}, null, Map.of());
  }

  @Test
  void derivesFromMetadataValueStr() {
    List<Document> docs =
        List.of(
            doc("d1", "label", new MetadataValue.Str("animal")),
            doc("d2", "label", new MetadataValue.Str("animal")),
            doc("d3", "label", new MetadataValue.Str("plant")));

    var result = MetadataQrelsDeriver.derive(docs, "label");

    assertThat(result.coverage().labeledDocs()).isEqualTo(3);
    assertThat(result.coverage().distinctLabels()).isEqualTo(2);
    assertThat(result.qrels().relevance().get("d1")).containsOnlyKeys("d2");
    assertThat(result.qrels().relevance().get("d2")).containsOnlyKeys("d1");
    assertThat(result.qrels().relevance().get("d3")).isNull();
  }

  @Test
  void derivesFromMetadataValueTags() {
    List<Document> docs =
        List.of(
            doc("d1", "tags", new MetadataValue.Tags(List.of("red", "small"))),
            doc("d2", "tags", new MetadataValue.Tags(List.of("red", "large"))),
            doc("d3", "tags", new MetadataValue.Tags(List.of("blue", "small"))));

    var result = MetadataQrelsDeriver.derive(docs, "tags");

    // d1 shares "red" with d2 and "small" with d3.
    assertThat(result.qrels().relevance().get("d1")).containsKeys("d2", "d3");
    assertThat(result.coverage().distinctLabels()).isEqualTo(4); // red, small, large, blue
  }

  @Test
  void failsFastWhenLabelCoverageLow() {
    List<Document> docs =
        List.of(
            doc("d1", "label", new MetadataValue.Str("a")),
            docNoMeta("d2"),
            docNoMeta("d3"),
            docNoMeta("d4"));
    assertThatThrownBy(() -> MetadataQrelsDeriver.derive(docs, "label"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("unlabeled");
  }

  @Test
  void reportsDistinctLabels() {
    List<Document> docs =
        List.of(
            doc("d1", "label", new MetadataValue.Str("a")),
            doc("d2", "label", new MetadataValue.Str("b")),
            doc("d3", "label", new MetadataValue.Str("c")),
            doc("d4", "label", new MetadataValue.Str("a")));

    var result = MetadataQrelsDeriver.derive(docs, "label");
    assertThat(result.coverage().distinctLabels()).isEqualTo(3);
    assertThat(result.coverage().labeledDocs()).isEqualTo(4);
    assertThat(result.coverage().unlabeledDocs()).isZero();
  }

  @Test
  void derivesFromOneShotIterable() {
    List<Document> docs =
        List.of(
            doc("d1", "label", new MetadataValue.Str("animal")),
            doc("d2", "label", new MetadataValue.Str("animal")),
            doc("d3", "label", new MetadataValue.Str("plant")));
    AtomicBoolean consumed = new AtomicBoolean();
    Iterable<Document> oneShot =
        () -> {
          if (!consumed.compareAndSet(false, true)) {
            throw new IllegalStateException("iterated twice");
          }
          return docs.iterator();
        };

    var result = MetadataQrelsDeriver.derive(oneShot, "label");

    assertThat(result.qrels().relevance().get("d1")).containsOnlyKeys("d2");
    assertThat(result.qrels().relevance().get("d2")).containsOnlyKeys("d1");
    assertThat(result.coverage().labeledDocs()).isEqualTo(3);
  }
}
