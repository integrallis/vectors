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
package com.integrallis.vectors.ingest.embedders;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ingest.IngestDoc;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PrecomputedEmbedderTest {

  private static IngestDoc precomputed(String id, float... v) {
    return new IngestDoc(id, "ignored", null, "text/plain", null, v);
  }

  @Test
  void returnsVectorsInInputOrder() {
    PrecomputedEmbedder e = new PrecomputedEmbedder(3);
    List<IngestDoc> docs =
        List.of(
            precomputed("a", 1f, 2f, 3f),
            precomputed("b", 4f, 5f, 6f),
            precomputed("c", 7f, 8f, 9f));
    List<float[]> vectors = e.embedAll(docs);
    assertThat(vectors).hasSize(3);
    assertThat(vectors.get(0)).containsExactly(1f, 2f, 3f);
    assertThat(vectors.get(1)).containsExactly(4f, 5f, 6f);
    assertThat(vectors.get(2)).containsExactly(7f, 8f, 9f);
  }

  @Test
  void exposesNameAndDimension() {
    PrecomputedEmbedder e = new PrecomputedEmbedder(64, "unit-test");
    assertThat(e.name()).isEqualTo("unit-test");
    assertThat(e.dimension()).isEqualTo(64);
  }

  @Test
  void singleEmbedDelegatesToBatch() {
    PrecomputedEmbedder e = new PrecomputedEmbedder(2);
    float[] v = e.embed(precomputed("x", 0.5f, -0.5f));
    assertThat(v).containsExactly(0.5f, -0.5f);
  }

  @Test
  void rejectsMissingPrecomputedVector() {
    PrecomputedEmbedder e = new PrecomputedEmbedder(3);
    IngestDoc bad = IngestDoc.text("noVec", "hello");
    assertThatThrownBy(() -> e.embedAll(List.of(bad)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("noVec")
        .hasMessageContaining("precomputedVector");
  }

  @Test
  void rejectsDimensionMismatch() {
    PrecomputedEmbedder e = new PrecomputedEmbedder(3);
    IngestDoc bad = precomputed("d", 1f, 2f);
    assertThatThrownBy(() -> e.embedAll(List.of(bad)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vector length 2")
        .hasMessageContaining("!= 3");
  }

  @Test
  void rejectsBadConstructorArgs() {
    assertThatThrownBy(() -> new PrecomputedEmbedder(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new PrecomputedEmbedder(-1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new PrecomputedEmbedder(3, " "))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
