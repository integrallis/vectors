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
package com.integrallis.vectors.db.metadata;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import com.integrallis.vectors.core.Document;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class InMemoryMetadataStoreTest {

  private static Document doc(String id) {
    return Document.of(id, new float[] {1f, 2f, 3f, 4f});
  }

  @Test
  void putAndGetRoundTrip() {
    InMemoryMetadataStore store = new InMemoryMetadataStore();
    store.put(0, doc("a"));
    store.put(1, doc("b"));

    assertThat(store.size()).isEqualTo(2);
    assertThat(store.get(0).id()).isEqualTo("a");
    assertThat(store.get(1).id()).isEqualTo("b");
  }

  @Test
  void copyOfProducesIndependentSuccessor() {
    InMemoryMetadataStore src = new InMemoryMetadataStore();
    src.put(0, doc("a"));
    src.put(1, doc("b"));

    InMemoryMetadataStore copy = InMemoryMetadataStore.copyOf(src);
    assertThat(copy.size()).isEqualTo(2);
    assertThat(copy.get(0).id()).isEqualTo("a");
    assertThat(copy.get(1).id()).isEqualTo("b");

    // Mutating the copy does not affect the source.
    copy.put(2, doc("c"));
    assertThat(copy.size()).isEqualTo(3);
    assertThat(src.size()).isEqualTo(2);
    assertThat(src.get(2)).isNull();

    // Deleting from the copy does not affect the source.
    copy.delete(0);
    assertThat(copy.get(0)).isNull();
    assertThat(src.get(0)).isNotNull();
  }

  @Test
  void copyOfEmptyStoreIsEmpty() {
    InMemoryMetadataStore copy = InMemoryMetadataStore.copyOf(new InMemoryMetadataStore());
    assertThat(copy.size()).isZero();
    copy.put(0, doc("a"));
    assertThat(copy.size()).isEqualTo(1);
  }

  @Test
  void copyOfNullThrows() {
    assertThatNullPointerException().isThrownBy(() -> InMemoryMetadataStore.copyOf(null));
  }
}
