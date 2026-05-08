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

import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IngestDocTest {

  @Test
  void textFactoryWiresDefaultMime() {
    IngestDoc d = IngestDoc.text("food-01", "sushi");
    assertThat(d.id()).isEqualTo("food-01");
    assertThat(d.text()).isEqualTo("sushi");
    assertThat(d.mime()).isEqualTo("text/plain");
    assertThat(d.attrs()).isEmpty();
    assertThat(d.precomputedVector()).isNull();
    assertThat(d.blob()).isNull();
  }

  @Test
  void rejectsBlankId() {
    assertThatThrownBy(() -> IngestDoc.text("  ", "x"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-blank");
  }

  @Test
  void rejectsDocWithoutPayload() {
    assertThatThrownBy(() -> new IngestDoc("id", null, null, null, null, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("text/blob/precomputedVector");
  }

  @Test
  void blobIsDefensivelyCopied() {
    byte[] src = {1, 2, 3};
    IngestDoc d = new IngestDoc("id", null, src, "application/octet-stream", null, null);
    src[0] = 99;
    assertThat(d.blob()).containsExactly(1, 2, 3);
    // Each accessor call returns a fresh copy.
    byte[] firstAccess = d.blob();
    firstAccess[0] = 99;
    assertThat(d.blob()).containsExactly(1, 2, 3);
  }

  @Test
  void precomputedVectorIsDefensivelyCopied() {
    float[] src = {1f, 2f};
    IngestDoc d = IngestDoc.precomputed("id", "x", src);
    src[0] = 99f;
    assertThat(d.precomputedVector()).containsExactly(1f, 2f);
    float[] copy = d.precomputedVector();
    copy[0] = 99f;
    assertThat(d.precomputedVector()).containsExactly(1f, 2f);
  }

  @Test
  void attrsBecomeImmutable() {
    IngestDoc d = new IngestDoc("id", "t", null, "text/plain", Map.of("k", "v"), null);
    assertThatThrownBy(() -> d.attrs().put("a", "b"))
        .isInstanceOf(UnsupportedOperationException.class);
  }
}
